import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import math
from timm.models import create_model
from transformers import ViTModel, ViTConfig
import torch.nn.functional as F
from random import randrange
from utils import append_to_log_file
import numpy as np


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
        self.original_embedding_dim = embed_dim
        self.patch_dim = patch_size * patch_size
        self.patch_embed = PatchEmbedding(img_height,img_width, patch_size, in_channels, embed_dim)


        n_patches = self.patch_embed.n_patches
        print("NPATCHES", n_patches)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #torch.zeroes(dimensione)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim)) #Ho completamente rimosso il cls token

        # Transformer Encoder Layers
        encoder_layer = TransformerEncoderLayerWithAttn(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio), #dim_feedforward è quanto aumenta d_model nel feedforward, qua fa da 768 a 4*768 e viceversa
                                                   activation='gelu',
                                                   batch_first=True,
                                                   dropout=0.1
                                                   )
        #self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        #self.encoder = encoder_layer
        self.encoders = nn.ModuleList([ encoder_layer for _ in range(depth) ]) #per il depth

        #Per pretrain
        self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, self.patch_dim))
        self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, self.patch_dim))
        self.mask_embed = torch.zeros(1, 1, embed_dim)
        #self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)
        self.norm = nn.LayerNorm(embed_dim)
        self.unfold = torch.nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
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

        for encoder in self.encoders:
            if attn:
                x, attn_weights = encoder(x, return_attn = True)  # [B, N+1, D]
            else:
                x = encoder(x)
        #x = self.norm(x)
        patch_tokens = x[:, 1:, :]                 # [B, N, D]
        pooled = patch_tokens.mean(dim=1)
        if attn:
            return pooled, attn_weights # spesso si prende x[:, 0] come rappresentazione globale (token CLS), prima riga per ogni batch
        else:
            return pooled
    
        

class MultiChannelViT(nn.Module):
    def __init__(self, n_channels=22, img_height=224, img_width = 224 ,patch_size=16,
                 embed_dim=768, num_classes=4, single = False, depth = 2, num_heads = 2):
        super().__init__()

        self.p_t_dim = img_width // patch_size
        self.original_embedding_dim = embed_dim
        self.patch_dim = patch_size*patch_size
        self.patch_size = patch_size
        self.img_height = img_height
        self.img_width = img_width

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
        

    def gen_maskid_patch(self, sequence_len=500, mask_size=2, cluster=3):
        """
        Genera indici casuali di patch da mascherare seguendo la logica originale SSAST.
        
        Args:
            sequence_len (int): numero totale di patch (es. 512)
            mask_size (int): numero di patch da mascherare
            p_t_dim (int): numero di patch sulla dimensione temporale (larghezza), es. 64 - 1008/16
            cluster (int): fattore massimo di clustering (cur_clus sarà tra 3 e 3+cluster-1)
        
        Returns:
            torch.LongTensor: indici delle patch mascherate, shape (mask_size,)
        """
        mask_id = []

        # randomizza clustering factor in [3, 3+cluster)
        cur_clus = randrange(cluster) + 3

        while len(set(mask_id)) <= mask_size:
            start_id = randrange(sequence_len) #sceglie una patch a caso tra le 512

            cur_mask = []
            for i in range(cur_clus):
                for j in range(cur_clus):
                    mask_cand = start_id + self.p_t_dim * i + j
                    if mask_cand >= 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)

            mask_id = mask_id + cur_mask

        # rimuove duplicati e limita al numero richiesto
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)
    

    def mpc(self, x, patch_embed, cpredlayer, mask_embed, num_patches, mask_patch, encoder, show_mask = False):
        """
        Masked Patch Pretraining (discriminative objective, InfoNCE)
        
        Args:
            x: input batch, shape [B, C, H, W]
            patch_embed: funzione / modulo che trasforma x in embedding patch, [B, N, D]
            cpredlayer: MLP per predizione patch, mappa da embedding a patch dim
            mask_embed: learnable mask embedding, shape [1, D]
            num_patches: numero totale di patch, 576
            mask_patch: numero di patch da mascherare
            encoder: encoder del singolo canale
            show_mask: solo per plottare
        
        Returns:
            acc: accuracy dei masked patch
            nce: InfoNCE loss
        """
        input = encoder.unfold(x).transpose(1, 2)
        B = x.shape[0]

        # passo 1: patch embedding
        x = patch_embed(x)  # [B, N, D]

        # inizializzo tensori per salvare i valori reali dei masked patch
        encode_samples = torch.empty((B, mask_patch, self.patch_dim), device=x.device, requires_grad=False).float()
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        mask_dense = torch.ones_like(x, device=x.device)

        # passo 2: genera indici dei patch da mascherare per ogni batch
        for i in range(B):
            mask_index[i] = self.gen_maskid_patch(sequence_len=num_patches, mask_size=mask_patch)

            # salva i valori reali dei patch mascherati
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            
            # Tiene conto di quali sono i patch da mascherare
            mask_dense[i, mask_index[i], :] = 0


        # passo 3: applica embedding del mask
        mask_tokens = mask_embed.to(x.device).expand(B, x.shape[1], -1)
        x = x * mask_dense + (1 - mask_dense) * mask_tokens

        # passo 4: passa attraverso il Transformer
        # aggiungi token CLS se il tuo modello li usa
        cls_token = encoder.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        #facciamo la media con pool

        # ipotizziamo che patch_embed abbia attribute `blocks` e `pos_embed`
        x = x + encoder.pos_embed

        for enc in encoder.encoders:
            x = enc(x)

        x = encoder.norm(x)

        # passo 5: predizione dei masked patch
        pred = torch.empty((B, mask_patch, self.patch_dim), device=x.device).float()
        for i in range(B):
            pred[i] = cpredlayer(x[i, mask_index[i] + 1, :])  # +1 per CLS token

        encode_samples = (encode_samples - encode_samples.mean(dim=-1, keepdim=True)) / (encode_samples.std(dim=-1, keepdim=True) + 1e-6)
        pred = (pred - pred.mean(dim=-1, keepdim=True)) / (pred.std(dim=-1, keepdim=True) + 1e-6)


        append_to_log_file("log.txt", f"encode_samples[{i}] mean: {encode_samples[i].mean().item()}, std: {encode_samples[i].std().item()}")
        append_to_log_file("log.txt", f"pred[{i}] mean: {pred[i].mean().item()}, std: {pred[i].std().item()}")

        # passo 6: calcolo InfoNCE loss
        softmax = nn.Softmax(dim=-1) #mettere dim 0?
        logsoftmax = nn.LogSoftmax(dim=-1)

        nce = torch.tensor(0., device=x.device)
        correct = torch.tensor(0., device=x.device)
        for i in np.arange(0,B):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # [mask_patch, mask_patch]
            correct += torch.sum(torch.eq(torch.argmax(softmax(total), dim=-1), torch.arange(0, mask_patch, device=x.device)))
            nce += torch.sum(torch.diag(logsoftmax(total)))

        acc = 1. * correct / (B * mask_patch)
        nce = nce / (-1. * B * mask_patch)



        
        if show_mask == False:
            return acc, nce
        else:
            if B > 1:
                raise Exception('Currently only support single spectrogram probing test.')
            self.mask_correct = torch.nn.Parameter(torch.arange(0, mask_patch), requires_grad=False)
            pred = input.clone()  # [B, 512, 256]
            masked = input.clone()
            for i in range(B):
                result = [float(t) * 99 for t in torch.eq(torch.argmax(softmax(total), dim=-1), self.mask_correct)]
                pred[i, mask_index[i], :] = torch.tensor(result).reshape(mask_patch, 1).expand(mask_patch, self.patch_dim)
                masked[i, mask_index[i], :] = 99.0
            fold = torch.nn.Fold(output_size=([self.img_height, self.img_width]), kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size)) ##
            pred = fold(pred.transpose(1, 2))
            masked = fold(masked.transpose(1, 2))
            return pred, masked
    
    def mpg(self, input, mask_patch, encoder, num_patches):
        B = input.shape[0]
        x = encoder.patch_embed(input)
        input = encoder.unfold(input).transpose(1,2)

        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            mask_index[i] = self.gen_maskid_patch(sequence_len=num_patches, mask_size=mask_patch)
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = encoder.mask_embed.to(x.device).expand(B, x.shape[1], -1)
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        cls_token = encoder.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        # go through the Transformer layers
        x = x + encoder.pos_embed
        #x = self.v.pos_drop(x) ######################
        for enc in encoder.encoders:
            x = enc(x)
        x = encoder.norm(x) #La normalizzazione è già presente nella classe encoder

        pred = torch.empty((B, mask_patch, self.patch_dim), device=x.device).float()  # e.g. size 12*100*256
        target = torch.empty((B, mask_patch, self.patch_dim), device=x.device).float()

        for i in range(B):
            pred[i] = encoder.gpredlayer(x[i, mask_index[i] + 1, :])
            target[i] = input[i, mask_index[i], :]

        # calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)

        return mse

    def plot_patches(self, x, mask_patch, show_mask = True, channel = 0):
        B, C, H, W = x.shape
        for i in range(channel, channel+1):
            encoder = self.encoders[channel]
            x_channel = x[:, i:i+1, :, :]  # [B, 1, H, W]
            x_channel = x_channel.transpose(2, 3)
            # chiamiamo la funzione mpc per il singolo canale
            pred, masked = self.mpc(
                x_channel,
                patch_embed=encoder.patch_embed,
                cpredlayer=encoder.cpredlayer,
                mask_embed=encoder.mask_embed,
                num_patches=encoder.patch_embed.n_patches,
                mask_patch=mask_patch,
                encoder = encoder,
                show_mask=True
            )
        return pred,masked

    def pret_forward(self, x, mask_patch):

        B, C, H, W = x.shape
        total_loss = 0
        total_acc = 0

        for i, encoder in enumerate(self.encoders):
            print(i)
            x_channel = x[:, i:i+1, :, :]  # [B, 1, H, W]
            # chiamiamo la funzione mpc per il singolo canale
            acc, nce = self.mpc(
                x_channel,
                patch_embed=encoder.patch_embed,
                cpredlayer=encoder.cpredlayer,
                mask_embed=encoder.mask_embed,
                num_patches=encoder.patch_embed.n_patches,
                mask_patch=mask_patch,
                encoder = encoder
            )

            mse_loss = self.mpg(x_channel, mask_patch=mask_patch, encoder=encoder, num_patches=encoder.patch_embed.n_patches)
            append_to_log_file("loss.txt", f"nce loss: {nce}, mse loss: {mse_loss}")
            acc = acc.mean()
            nce = nce.mean()
            mse_loss = mse_loss.mean()
            total_loss += nce + 4*mse_loss
            total_acc += acc
            #append_to_log_file("/mnt/localstorage/cdeangelis/Multi_checkpoints/MVIT_DEF_NOPRETRAIN35/log_single_config_MVIT_DEF_NOPRETRAIN35.txt", f"Channel {i}, nce loss: {nce}, mse loss: {mse_loss}, acc: {acc}")

        # media sulle canali
        total_loss = total_loss / C
        total_acc = total_acc / C

        return total_loss, total_acc, mse_loss, nce

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
            return out