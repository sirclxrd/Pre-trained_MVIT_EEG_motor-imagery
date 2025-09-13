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

class EEGSpatialAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, dropout=0.3):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final linear projection after concat heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Normalization and dropout
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Feed-forward layer (position-wise MLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, D] (N = 22 canali)
        B, N, D = x.shape

        # Linear projections
        Q = self.q_proj(x)  # [B, N, D]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, h, N, N]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [B, h, N, d]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]

        # Output projection
        x = x + self.dropout(self.out_proj(attn_output))  # residual connection
        x = self.ln1(x)

        # Feed-forward
        x = x + self.mlp(x)  # residual connection
        x = self.ln2(x)

        # Pooling (mean over channels)
        x = x.mean(dim=1)  # [B, D]

        return x  # puoi passarlo a un classificatore



#da 0 a 3.996 ho (22,1000)
#da 0 a 4.028 o (22,1008) buono per fare 16x16 patch

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
        #stride = patch_size//2 
        #padding = 0
        #self.n_patches = int(((img_height - patch_size + 2 * padding) // stride + 1) * \
        #                ((img_width - patch_size + 2 * padding) // stride + 1)) #patch overlap

        patch_dim = in_channels * patch_height * patch_width
        

        # cnn_name='resnet34'
        # self.cnn = create_model(cnn_name, pretrained=False, features_only=True, in_chans=in_channels)
        # self.cnn_out_dim = self.cnn.feature_info[-1]['num_chs']  # typically 512 or 2048


        #self.conv_proj = nn.Conv2d(self.cnn_out_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.conv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width) ) 

        self.norm = nn.LayerNorm(embed_dim)
        
        #proiezione come nel paper originale con flatten
        # (h ph) specificando ph e pw signfica fare h = (h / ph)
        # alla fine ottengo [b, n_patches, dim_patch]

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! HO COMMENTATO QUESTO PER RISPARMIARE MEMORIA
        # self.vit_proj = nn.Sequential(
        #     Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', 
        #               ph=patch_height, pw=patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, embed_dim),
        #     nn.LayerNorm(embed_dim)
        # )


    def forward(self, x):
        # x: [B, 3, 224, 224] -> [B, 768, 14, 14] -> flatten
        if self.withconv == True:
            #x = self.conv_proj1(x)
            #x = self.conv_proj2(x)
            #x = self.cnn(x)[-1]
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
        #self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder = nn.ModuleList([
            encoder_layer for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, attn = False):
        x = self.patch_embed(x)  # [B, N, D]
        B, N, D = x.shape

        # Aggiunta del token CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]
        #print("X",x.shape)
        #print("Pos",self.pos_embed.shape)
        x = x + self.pos_embed  # aggiunta positional embedding
        # print("After patch layer shape: ", x.shape)
        for encoder in self.encoder:
            if attn:
                x, attn_weights = encoder(x, return_attn = True)  # [B, N+1, D]
            else:
                x = encoder(x)
            x = self.norm(x)

        if attn:
            return x[:,0], attn_weights # spesso si prende x[:, 0] come rappresentazione globale (token CLS), prima riga per ogni batch
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
                nn.Linear(embed_dim * n_channels, 4)
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
        #self.eeg_attention = EEGSpatialAttention(embed_dim, num_heads, 0.3)
        # last_transformer = nn.TransformerEncoderLayer(d_model=embed_dim,
        #                                            nhead=num_heads,
        #                                            dim_feedforward=int(embed_dim * 4), #dim_feedforward è quanto aumenta d_model nel feedforward, qua fa da 768 a 4*768 e viceversa
        #                                            activation='gelu',
        #                                            batch_first=True,
        #                                            dropout=0.2
        #                                            )
        # self.last_encoder = nn.TransformerEncoder(last_transformer, num_layers=depth)
        # self.norm = nn.LayerNorm(embed_dim)

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
                
                #nel caso voglio controllare gli output dei singoli canali
                #c_out = self.single_classifier(token)
                #channels.append(c_out)

                tokens.append(token)
            concat_token = torch.cat(tokens, dim=-1)  # [B, 22*D]
            out = self.concat_classifier(concat_token)

            # tokens = torch.stack(tokens, dim=1)
            # attn_output = self.eeg_attention(tokens)
            # out = self.single_classifier(attn_output)

            # tokens = torch.stack(tokens, dim=1)  # [B, 22, D]
            # attn_output = self.last_encoder(tokens)         # [B, D]
            # attn_output = attn_output.mean(dim=1)  # [B, D]
            # attn_output = self.norm(attn_output)
            # out = self.single_classifier(attn_output)
        # SINGLE VIT
        else:
            single_token = self.encoder(x)
            # print(single_token.shape)
            out = self.single_classifier(single_token)      # [B, num_classes]

        if attn:
            return out,attn_weights, concat_token
        else:
            return out, concat_token


class ViTEncoderEEG(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # 1. Creo config identica a vit-base
        config = ViTConfig.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        config.num_channels = 1  # input a 1 canale
        
        # 2. Creo il modello da config
        self.vit = ViTModel(config)
        
        if pretrained:
            # Carico il modello pretrained per copiare i pesi
            pretrained_model = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
            with torch.no_grad():
                # media dei pesi RGB del patch embedding
                w = pretrained_model.embeddings.patch_embeddings.projection.weight  # [hidden,3,P,P]
                self.vit.embeddings.patch_embeddings.projection.weight[:] = w.mean(dim=1, keepdim=True)
                # copia bias se esiste
                if pretrained_model.embeddings.patch_embeddings.projection.bias is not None:
                    self.vit.embeddings.patch_embeddings.projection.bias[:] = pretrained_model.embeddings.patch_embeddings.projection.bias

    def forward(self, x):
        """
        x: [B, 1, 32, 1008]
        """
        # Padding per rendere "quasi quadrato"
        B, C, H, W = x.shape
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        
        # Passaggio nel ViT
        outputs = self.vit(pixel_values=x)
        # CLS token come embedding globale
        return outputs.last_hidden_state[:,0]  # [B, hidden_dim]


# model = MultiChannelViT(n_channels=22, img_height = 32, img_width = 1008, patch_size=16, embed_dim=768, num_classes=4, single=False)
# criterion = nn.CrossEntropyLoss() #contiene già una softmax
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
# dummy_input = torch.randn(8, 22, 32, 1008)  # 8 esempi, 22 canali, 32x32
# output = model(dummy_input)  # [8, 4]
# print(output)
