import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import math
from timm.models import create_model
from transformers import ViTModel, ViTConfig
import torch.nn.functional as F

class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attn=False):
        # src: [B, N, D]
        q = k = src
        attn_output, attn_weights = self.self_attn(
            q, k, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # [B, num_heads, N, N]
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        if return_attn:
            return src, attn_weights
        else:
            return src


class PatchEmbedding(nn.Module):
    def __init__(self, img_height=224, img_width = 224,patch_size=16, in_channels=3, embed_dim=768, withconv = True):
        super().__init__()

        height, width = img_height, img_width
        patch_height, patch_width = patch_size, patch_size #########
        self.patch_size = patch_size
        self.withconv = withconv

        if self.withconv:
            print("You are using the CONV patch embedding")
        else:
            print("You are using the original VIT patch embedding")

        assert height % patch_height == 0 and width % patch_width == 0, \
            "Le dimensioni dell'immagine devono essere divisibili per la patch size"


        self.n_patches = (height // patch_height) * (width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        self.conv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width) ) 
        self.norm = nn.LayerNorm(embed_dim)



    def forward(self, x):
        # x: [B, 3, 224, 224] -> [B, 768, 14, 14] -> flatten
        if self.withconv == True:
            x = self.conv_proj(x)  # [B, embed_dim, H', W']
            x = x.flatten(2)  # [B, embed_dim, N]
            x = x.transpose(1, 2)  # [B, N, embed_dim]
            x = self.norm(x)
        else:
            x = self.vit_proj(x)
        return x
    
        

class ViTEncoder(nn.Module):
    def __init__(self, img_height=224, img_width=224 ,patch_size=16, in_channels=3,
                 embed_dim=768, depth=2, num_heads=2, mlp_ratio=2.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_height,img_width, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        print("NPATCHES", n_patches)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #torch.zeroes(dimensione)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        # Transformer Encoder Layers
        encoder_layer = TransformerEncoderLayerWithAttn(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio), #dim_feedforward è quanto aumenta d_model nel feedforward, qua fa da 768 a 4*768 e viceversa
                                                   activation='gelu',
                                                   batch_first=True,
                                                   dropout=0.5
                                                   )
        self.encoder = nn.ModuleList([
            encoder_layer for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, attn = False):
        x = self.patch_embed(x)  # [B, N, D]
        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]
        x = x + self.pos_embed  #positional embedding
        for encoder in self.encoder:
            if attn:
                x, attn_weights = encoder(x, return_attn = True)  # [B, N+1, D]
            else:
                x = encoder(x)
            x = self.norm(x)

        if attn:
            return x[:,0], attn_weights # or x[:,0]
        else:
            return x[:,0]

class MultiChannelViT(nn.Module):
    def __init__(self, n_channels=22, img_height=224, img_width = 224 ,patch_size=16,
                 embed_dim=768, num_classes=4, single = False, depth = 2, num_heads = 2):
        super().__init__()
        if single == False:
            print("You are using MVIT")
        else:
            print("You are using classic VIT")
        if single == False:
            self.encoders = nn.ModuleList([
                ViTEncoder(img_height=img_height,
                           img_width = img_width,
                        patch_size=patch_size,
                        in_channels=1,
                        embed_dim=embed_dim)
                        for _ in range(n_channels)
            ])
            

            # classifier per output concatenato
            self.concat_classifier = nn.Sequential(
                nn.Linear(embed_dim * n_channels, num_classes)
            )

        else:
            self.encoder = ViTEncoder(img_height=img_height,
                        img_width = img_width,
                        patch_size=patch_size,
                        in_channels=n_channels,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads)

            # classifier per output singolo
            self.single_classifier = nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5), ######
                nn.Linear(512, num_classes)
            )
            

        self.single = single


    def forward(self, x, attn = False):
        # in questo modo devo dare in input tutti gli spettrogrammi concatenati sulla profondità
        # x: [B, C, H, W] = [B, 22, 32, 32]

        # MVIT
        if self.single == False:
            tokens = []
            channels = []
            for i, encoder in enumerate(self.encoders):
                channel_i = x[:, i:i+1, :, :]  # [B, 1, H, W]

                if attn:
                    token, attn_weights = encoder(channel_i, True)     # [B, D]
                else:
                    token = encoder(channel_i)
                tokens.append(token)
            concat_token = torch.cat(tokens, dim=-1)  # [B, 22*D]
            out = self.concat_classifier(concat_token)
        # SINGLE VIT
        else:
            single_token = self.encoder(x)
            # print(single_token.shape)
            out = self.single_classifier(single_token)      # [B, num_classes]
            concat_token = single_token
        if attn:
            return out,attn_weights, concat_token
        else:
            return out, concat_token





