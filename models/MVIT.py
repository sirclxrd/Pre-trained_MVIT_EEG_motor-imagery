import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import math
from timm.models import create_model
import torch.nn.functional as F



class PatchEmbedding(nn.Module):
    def __init__(self, img_height=224, img_width = 224,patch_size=16, in_channels=3, embed_dim=768, withconv = True, patch_width = 16):
        super().__init__()

        height, width = img_height, img_width
        patch_height, patch_width = patch_size, patch_width #########
        self.patch_size = patch_size
        self.withconv = withconv

        if self.withconv:
            print("You are using the CONV patch embedding")
        else:
            print("You are using the original VIT patch embedding")

        assert height % patch_height == 0 and width % patch_width == 0, \
            f"Le dimensioni dell'immagine devono essere divisibili per la patch size {img_height} {img_width}"


        self.n_patches = (height // patch_height) * (width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        self.conv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width) ) 

        self.norm = nn.LayerNorm(embed_dim)
        self.vit_proj = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', 
                      ph=patch_height, pw=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )


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
                 embed_dim=768, depth=2, num_heads=2, mlp_ratio=2.0, patch_width=16):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_height,img_width, patch_size, in_channels, embed_dim, patch_width=patch_width)
        self.img_size = (img_height, img_width)
        self.patch_size = patch_size
        n_patches = self.patch_embed.n_patches
        print("NPATCHES", n_patches)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #torch.zeroes(dimensione)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))


        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio), #dim_feedforward è quanto aumenta d_model nel feedforward, qua fa da 768 a 4*768 e viceversa
                                                   activation='gelu',
                                                   batch_first=True,
                                                   dropout=0.5
                                                   )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, return_patches = False):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, N, D]
        B, N, D = x.shape

        # Aggiunta del token CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]
        x = x + self.pos_embed  # aggiunta positional embedding
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        x = self.encoder(x)  # [B, N+1, D]
        x = self.norm(x)

        if return_patches:
            return x[:, 1:, :] 
        else:
            patch_tokens = x[:, 1:, :]                 # [B, N, D]
            pooled = patch_tokens.mean(dim=1)          # [B, D]           
            return pooled

    def interpolate_pos_embed(self, x, pos_embed, grid_size_hw):
        """
        Interpola il positional embedding per adattarsi a input di dimensione variabile (anche rettangolare).
        - x: tensor [B, N, D]
        - pos_embed: tensor [1, N_orig, D]
        - grid_size_hw: tuple (H, W) dimensione griglia di patch corrente
        """
        B, N, D = x.shape
        cls_token = pos_embed[:, :1]       # [1, 1, D]
        patch_pos_embed = pos_embed[:, 1:] # [1, N-1, D]

        num_patches = patch_pos_embed.shape[1]
        orig_h = self.img_size[0] // self.patch_size
        orig_w = self.img_size[1] // self.patch_size

        patch_pos_embed = patch_pos_embed.reshape(1, orig_h, orig_w, D).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, 
            size=grid_size_hw, 
            mode='bilinear', 
            align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, grid_size_hw[0] * grid_size_hw[1], D)

        new_pos_embed = torch.cat((cls_token, patch_pos_embed), dim=1)
        return new_pos_embed

class MultiChannelViT(nn.Module):
    def __init__(self, n_channels=22, img_height=224, img_width = 224 ,patch_size=16,
                 embed_dim=768, num_classes=4, single = False, depth = 2, num_heads = 2):
        super().__init__()
        if single == False:
            print("You are using MVIT")
        else:
            print("You are using classic VIT")
        if single == False:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #torch.zeroes(dimensione)
            self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)


            self.encoders = nn.ModuleList([
                ViTEncoder(img_height=img_height,
                           img_width = img_width,
                        patch_size=patch_size,
                        patch_width = patch_size, ##########
                        in_channels=1,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads = num_heads)
                for _ in range(math.ceil(n_channels))
            ])
        else:
            self.encoder = ViTEncoder(img_height=img_height,
                        img_width = img_width,
                        patch_size=patch_size,
                        in_channels=n_channels,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads)
        
        self.first = embed_dim * math.ceil(n_channels)
        self.concat_classifier = nn.Sequential(
            nn.Linear(embed_dim*n_channels, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        self.single_classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5), ######
            nn.Linear(512, num_classes)
        )
        self.single = single
        last_transformer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * 2), #dim_feedforward è quanto aumenta d_model nel feedforward, qua fa da 768 a 4*768 e viceversa
                                                   activation='gelu',
                                                   batch_first=True,
                                                   dropout=0.5
                                                   )
        self.encoder = nn.TransformerEncoder(last_transformer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        

    def forward(self, x, ret_concat = False):
        # MVIT
        if self.single == False:
            tokens = []
            channels = []
            for i, encoder in enumerate(self.encoders):
                channel_i = x[:, i:i+1, :, :]  # [B, 1, H, W]
                token = encoder.patch_embed(channel_i)
                tokens.append(token)
            x = torch.cat(tokens, dim=1)  # [B, 22*D]

            B, N, D = x.shape
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
            x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]
            pos_embed = self.pos_embed.expand(B, -1, -1)
            x = x + self.pos_embed  # aggiunta positional embedding
            x = self.encoder(x)  # [B, N+1, D]
            x = self.norm(x)

            patch_tokens = x[:, 1:, :]                 # [B, N, D]
            pooled = patch_tokens.mean(dim=1) 

            out = self.single_classifier(pooled)
        else:
            single_token = self.encoder(x)
            out = self.single_classifier(single_token)      # [B, num_classes]

        if ret_concat:
            return out, pooled
        else:
            return out







