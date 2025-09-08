import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import math
from timm.models import create_model
import torch.nn.functional as F



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
            "Le dimensioni dell'immagine devono essere divisibili per la patch size"

        #self.pre_pool = nn.AvgPool2d(kernel_size=(1,2), stride=(1,2))
        #width = width // 2

        self.n_patches = (height // patch_height) * (width // patch_width)
        #stride = patch_size//2 
        #padding = 0
        #self.n_patches = int(((img_height - patch_size + 2 * padding) // stride + 1) * \
        #                ((img_width - patch_size + 2 * padding) // stride + 1)) #patch overlap
        
        # reduced_channels = 6
        # self.bottleneck = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        # in_channels = reduced_channels

        patch_dim = in_channels * patch_height * patch_width
        

        # cnn_name='resnet34'
        # self.cnn = create_model(cnn_name, pretrained=False, features_only=True, in_chans=in_channels)
        # self.cnn_out_dim = self.cnn.feature_info[-1]['num_chs']  # typically 512 or 2048

        #self.conv_proj = nn.Conv2d(self.cnn_out_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.conv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width) ) 

        self.embedding = nn.Sequential(
            nn.BatchNorm2d(embed_dim),     # normalizzazione spaziale
            nn.GELU(),                     # non linearità più smooth di ReLU
            nn.Dropout2d(0.2)              # regolarizzazione leggera
            # opzionale: nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.norm = nn.LayerNorm(embed_dim)
        # self.bn = nn.BatchNorm2d(embed_dim)
        # self.act = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(0.5)
        
        #proiezione come nel paper originale con flatten
        # (h ph) specificando ph e pw signfica fare h = (h / ph)
        # alla fine ottengo [b, n_patches, dim_patch]
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
            #x = self.conv_proj1(x)
            #x = self.conv_proj2(x)
            #x = self.cnn(x)[-1]
            #x = self.bottleneck(x)
            #x = self.pre_pool(x)
            x = self.conv_proj(x)  # [B, embed_dim, H', W']
            #x = self.embedding(x)
            x = x.flatten(2)  # [B, embed_dim, N]
            x = x.transpose(1, 2)  # [B, N, embed_dim]
            x = self.norm(x)
            # x = self.bn(x)
            # x = self.act(x)
            # x = self.dropout(x)
            #print(x.shape)
        else:
            x = self.vit_proj(x)
        return x

class PatchEmbedding2(nn.Module):
    """
    Factorized Convolutional Stem Design (2+1)D per EEG spettrogramma
    Input: [B, C=22, H=32, W=1008]
    Output: [B, N, embed_dim], dove N = numero totale di patch
    """

    def __init__(self,
                 in_channels=22,       # numero di canali EEG
                 embed_dim=768,        # dimensione embedding finale
                 f=64,                 # numero canali intermedi dopo conv2D
                 H=32,                 # altezza spettrogramma
                 W=1008,               # lunghezza temporale
                 patch_height=4,       # altezza patch spaziale
                 patch_width=32,       # larghezza patch temporale
                 dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.f = f

        # aggiungiamo dimensione temporale T=1 per lo spettrogramma singolo
        T = 1
        patch_time = 1  # kernel temporale = 1 perché T=1

        # Calcolo numero patch
        H_p = H // patch_height
        W_p = W // patch_width
        T_p = T // patch_time
        self.n_patches = H_p * W_p * T_p

        # --- Convoluzione spaziale 2D (sui bins di frequenza e tempo) ---
        self.conv2d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=f,
            kernel_size=(patch_time, patch_height, patch_width),
            stride=(patch_time, patch_height, patch_width)
        )
        self.bn2d = nn.BatchNorm3d(f)
        self.act = nn.ReLU(inplace=True)

        # --- Convoluzione 1D temporale ---
        self.conv1d = nn.Conv3d(
            in_channels=f,
            out_channels=embed_dim,
            kernel_size=(patch_time, 1, 1),
            stride=(patch_time, 1, 1)
        )
        self.bn1d = nn.BatchNorm3d(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        # aggiungi dimensione temporale T=1
        x = x.unsqueeze(2)  # [B, C, T=1, H, W]

        # conv2D spaziale
        x = self.conv2d(x)
        x = self.bn2d(x)
        x = self.act(x)

        # conv1D temporale
        x = self.conv1d(x)
        x = self.bn1d(x)
        x = self.act(x)
        x = self.dropout(x)

        # reshape in sequenza di token [B, N, d]
        B, d, T_p, H_p, W_p = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).reshape(B, self.n_patches, d)

        return tokens

    
        

class ViTEncoder(nn.Module):
    def __init__(self, img_height=224, img_width=224 ,patch_size=16, in_channels=3,
                 embed_dim=768, depth=2, num_heads=2, mlp_ratio=2.0, patch_width=16):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_height,img_width, patch_size, in_channels, embed_dim, patch_width=patch_width)
        #self.patch_embed = PatchEmbedding2()
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
        #x = x + self.interpolate_pos_embed(x, self.pos_embed, (grid_h, grid_w))
        # print("After patch layer shape: ", x.shape)
        x = self.encoder(x)  # [B, N+1, D]
        x = self.norm(x)

        # spesso si prende x[:, 0] come rappresentazione globale (token CLS), prima riga per ogni batch
        if return_patches:
            return x[:, 1:, :]  # escludo CLS, patch embeddings
        else:
            patch_tokens = x[:, 1:, :]                 # [B, N, D]
            pooled = patch_tokens.mean(dim=1)          # [B, D]
            
            # Opzionale: concat CLS + pooled
            # pooled = torch.cat([x[:, 0], pooled], dim=-1)  # [B, 2D]
            
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

        # Dimensione originale della griglia (da pretraining)
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

        # Ricompone con CLS token
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
            #self.encoders = nn.ModuleList([
                # ViTEncoder(img_height=img_height,
                #            img_width = img_width,
                #         patch_size=patch_size,
                #         in_channels=1,
                #         embed_dim=embed_dim)

            #2a
            param_sets = [
            dict(patch_size=1,  patch_width = 336,  embed_dim=embed_dim, depth=depth, num_heads=num_heads),
            dict(patch_size=32, patch_width = 1, embed_dim=embed_dim, depth=depth, num_heads=num_heads),
            dict(patch_size=16, patch_width = 8, embed_dim=embed_dim, depth=depth, num_heads=num_heads)
            ]

            #2b
            # param_sets = [
            # dict(patch_size=1,  patch_width = 142,  embed_dim=embed_dim, depth=depth, num_heads=num_heads),
            # dict(patch_size=32, patch_width = 1, embed_dim=embed_dim, depth=depth, num_heads=num_heads),
            # dict(patch_size=16, patch_width = 8, embed_dim=embed_dim, depth=depth, num_heads=num_heads)
            # ]

            #physionet
            # param_sets = [
            # dict(patch_size=1,  patch_width = 160,  embed_dim=embed_dim, depth=depth, num_heads=num_heads),
            # dict(patch_size=32, patch_width = 1, embed_dim=embed_dim, depth=depth, num_heads=num_heads),
            # dict(patch_size=16, patch_width = 8, embed_dim=embed_dim, depth=depth, num_heads=num_heads)
            # ]

            self.encoders = nn.ModuleList([
                ViTEncoder(
                    img_height=img_height,
                    img_width=img_width,
                    in_channels=n_channels,
                    **params
                )
                for params in param_sets
            ])
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #torch.zeroes(dimensione)
            self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)


            # self.encoders = nn.ModuleList([
            #     ViTEncoder(img_height=img_height,
            #                img_width = img_width,
            #             patch_size=patch_size,
            #             in_channels=1,
            #             embed_dim=embed_dim,
            #             depth=depth,
            #             num_heads = num_heads)
            #     for _ in range(math.ceil(n_channels)) #22/3 lo arrotonda come se fosse 24/3
            # ])
        else:
            self.encoder = ViTEncoder(img_height=img_height,
                        img_width = img_width,
                        patch_size=patch_size,
                        in_channels=n_channels,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads)
        
        self.first = embed_dim * math.ceil(n_channels)
        # classifier per output concatenato
        self.concat_classifier = nn.Sequential(
            nn.Linear(embed_dim*len(param_sets), 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # classifier per output singolo
        self.single_classifier2 = nn.Sequential(
            nn.Linear(embed_dim*2, 512),
            nn.ReLU(),
            nn.Dropout(0.5), ######
            nn.Linear(512, num_classes)
        )

        self.single_classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5), ######
            nn.Linear(512, num_classes)
        )
        self.single = single
        #self.eeg_attention = EEGSpatialAttention(embed_dim, num_heads, 0.3)
        last_transformer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * 2), #dim_feedforward è quanto aumenta d_model nel feedforward, qua fa da 768 a 4*768 e viceversa
                                                   activation='gelu',
                                                   batch_first=True,
                                                   dropout=0.5
                                                   )
        self.encoder = nn.TransformerEncoder(last_transformer, num_layers=depth)
        #self.encoder = PretrainedViTEncoder()
        self.norm = nn.LayerNorm(embed_dim)

        

    def forward(self, x, group = False):
        # in questo modo devo dare in input tutti gli spettrogrammi concatenati sulla profondità
        # x: [B, C, H, W] = [B, 22, 32, 32]

        # MVIT
        if self.single == False and group == False:
            tokens = []
            channels = []
            for i, encoder in enumerate(self.encoders):
                #channel_i = x[:, i:i+1, :, :]  # [B, 1, H, W]
                #token = encoder.patch_embed(channel_i)
                token = encoder.patch_embed(x)     # [B, D]
                
                #nel caso voglio controllare gli output dei singoli canali
                #c_out = self.single_classifier(token)
                #channels.append(c_out)

                tokens.append(token)
            x = torch.cat(tokens, dim=1)  # [B, 22*D]

            x2 = x.mean(dim=1)
            out2 = self.single_classifier(x2)

            B, N, D = x.shape
            # Aggiunta del token CLS
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
            x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]
            pos_embed = self.pos_embed.expand(B, -1, -1)
            x = x + self.pos_embed  # aggiunta positional embedding
            #x = x + self.interpolate_pos_embed(x, self.pos_embed, (grid_h, grid_w))
            # print("After patch layer shape: ", x.shape)
            x = self.encoder(x)  # [B, N+1, D]
            x = self.norm(x)

            cls_rep = x[:, 0]
            patch_tokens = x[:, 1:, :]                 # [B, N, D]
            pooled = patch_tokens.mean(dim=1) 
            out = torch.cat([cls_rep, pooled], dim=-1)

            out = self.single_classifier2(out)

            # tokens = torch.stack(tokens, dim=1)
            # attn_output = self.eeg_attention(tokens)
            # out = self.single_classifier(attn_output)
        # SINGLE VIT
        elif self.single == False and group == True:
            tokens = []
            channels = []
            n_channels = x.shape[1]
            step = 3
            for i, encoder in enumerate(self.encoders):
                start_idx = i * step
                end_idx = start_idx + step
                
                # Prendo i canali da start_idx a end_idx
                channel_i = x[:, start_idx:end_idx, :, :]  # [B, fino a 3, H, W]
                
                # Se ultimi canali < 3, replico l'ultimo canale
                if channel_i.shape[1] < step:
                    last_channel = channel_i[:, -1:, :, :]
                    reps = step - channel_i.shape[1]
                    extra = last_channel.repeat(1, reps, 1, 1)
                    channel_i = torch.cat([channel_i, extra], dim=1) 
                
                token = encoder(channel_i)  # encoder si aspetta input con 3 canali
                tokens.append(token)
            concat_token = torch.cat(tokens, dim=-1)  # concatena su dimensione embed
            out = self.concat_classifier(concat_token)
        else:
            single_token = self.encoder(x)
            # print(single_token.shape)
            out = self.single_classifier(single_token)      # [B, num_classes]

        return out, out2


class MultiChannelViTSelfSupervised(nn.Module):
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
        else:
            self.encoder = ViTEncoder(img_height=img_height,
                        img_width = img_width,
                        patch_size=patch_size,
                        in_channels=n_channels,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads)
            
        # classifier per output singolo
        hidden_dim = 384
        out_dim = 192
        self.single_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.single = single

    def forward(self, x):
        # in questo modo devo dare in input tutti gli spettrogrammi concatenati sulla profondità
        # x: [B, C, H, W] = [B, 22, 32, 32]

        single_token = self.encoder(x)
        # print(single_token.shape)
        out = self.single_classifier(single_token)      
        out = nn.functional.normalize(out, dim=-1)


        return out

class RelativeLocalizationLoss(nn.Module):
    def __init__(self, embed_dim, grid_shape, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.grid_height, self.grid_width = grid_shape  # (2, 63)
        self.l1_loss = nn.L1Loss()

    def forward(self, patch_embeddings):
        # patch_embeddings: [B, N, D] where N = H*W
        B, N, D = patch_embeddings.shape
        H, W = self.grid_height, self.grid_width
        assert N == H * W, f"Expected {H*W} patches, got {N}"

        # Precompute coordinate grid: (N, 2)
        coords = torch.tensor([
            (i, j) for i in range(H) for j in range(W)
        ], device=patch_embeddings.device).float()

        losses = []
        for b in range(B):
            emb = patch_embeddings[b]  # [N, D]

            # Sample m distinct pairs
            m = min(100, N*(N-1)//2)
            idx1 = torch.randint(0, N, (m,), device=emb.device)
            idx2 = torch.randint(0, N, (m,), device=emb.device)

            # Ensure idx1 ≠ idx2
            mask = idx1 != idx2
            idx1, idx2 = idx1[mask], idx2[mask]
            if len(idx1) == 0: continue  # skip if all equal
            e1 = emb[idx1]  # [m', D]
            e2 = emb[idx2]  # [m', D]

            coord1 = coords[idx1]  # [m', 2]
            coord2 = coords[idx2]  # [m', 2]

            # Compute normalized distance in [0, 1]
            tu = torch.abs(coord1[:, 0] - coord2[:, 0]) / H
            tv = torch.abs(coord1[:, 1] - coord2[:, 1]) / W
            target = torch.stack([tu, tv], dim=1)  # [m', 2]

            inp = torch.cat([e1, e2], dim=1)  # [m', 2*D]
            pred = self.mlp(inp)  # [m', 2]

            loss = self.l1_loss(pred, target)
            losses.append(loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=patch_embeddings.device)

class ViTEncoderPretrained(nn.Module):
    def __init__(self, img_height, img_width, patch_size, in_channels, embed_dim,
                 depth=12, num_heads=12, pretrained=True):
        super().__init__()
        
        # Crea un ViT pre-addestrato su ImageNet
        model = create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            img_size=(img_height, img_width),
            in_chans=in_channels,
        )
        
        # Se in_channels ≠ 3, inizializza il primo layer con pesi adattati
        if pretrained and in_channels != 3:
            old_conv = create_model('vit_small_patch16_224', pretrained=True).patch_embed.proj
            new_conv = model.patch_embed.proj
            with torch.no_grad():
                if in_channels == 1:
                    mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight.copy_(mean_weight)
                else:
                    repeat = (in_channels // 3) + 1
                    expanded = old_conv.weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
                    new_conv.weight.copy_(expanded)
            model.patch_embed.proj = new_conv
        
        # Rimuove la classification head, lasciando solo il backbone
        model.head = nn.Identity()
        
        self.vit = model

    def forward(self, x):
        return self.vit(x)  # restituisce il token CLS

class PretrainedViTEncoder(nn.Module):
    def __init__(self, model_name='vit_small_patch16_224', pretrained=True):
        super().__init__()
        # carico un ViT pretrainato
        vit = create_model(model_name, pretrained=pretrained)

        # prendo solo l'encoder (blocchi Transformer + norm finale)
        self.encoder = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

    def forward(self, x):
        """
        x: [B, N, D] dove D deve essere self.embed_dim (768 nel caso base).
        """
        h = x
        for blk in self.encoder:
            h = blk(h)
        h = self.norm(h)  # [B, N, D]
        return h



# model = MultiChannelViT(n_channels=22, img_height = 32, img_width = 1008, patch_size=16, embed_dim=768, num_classes=4, single=False)
# criterion = nn.CrossEntropyLoss() #contiene già una softmax
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
# dummy_input = torch.randn(8, 22, 32, 1008)  # 8 esempi, 22 canali, 32x32
# output = model(dummy_input)  # [8, 4]
# print(output)
