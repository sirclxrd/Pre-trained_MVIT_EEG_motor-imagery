import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from timm.layers import to_2tuple,trunc_normal_

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()

        #768 è 16*16*3
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2)  # [B, embed_dim, N]
        x = x.transpose(1, 2)  # [B, N, embed_dim]
        return x
    

class pret_VIT(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 4 for Bci2a
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=4, fstride=16, tstride=16, input_fdim=32, input_tdim=1008, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True, single = False):
        super(pret_VIT, self).__init__()
        #assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'
        #timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if model_size == 'tiny224':
            self.v = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'small224':
            self.v = timm.create_model('deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base224':
            self.v = timm.create_model('deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base384':
            self.v = timm.create_model('deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')

        self.single = single
        self.original_num_patches = self.v.patch_embed.num_patches#196 
        self.oringal_hw = int(self.original_num_patches ** 0.5) #numero di patch 14x14
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim) #fstride e tstride dim patch, inputfdim e tdim dim immagine
        num_patches = f_dim * t_dim #restituisce tipo 14x14
        self.v.patch_embed.num_patches = num_patches

        #Inizializzo il mio livello di patch embedding per sostituirlo con quello del vit pretreinato per dare in input img di ogni dimensione
        #Ma prima faccio una somma dei pesi per non perderli
        self.mypatch_embed = PatchEmbed()

        if single == False:
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        else:
            n_channels = 22
            new_proj = torch.nn.Conv2d(n_channels, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))

        if imagenet_pretrain == True:
            print(self.v.patch_embed.proj.weight.shape) #[768, 3 , 16, 16]
            if single == False:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)) #fa la somma dei valori dei pesi siccome il patch embedding del vit come dimensione 1 ha i 3 channels
            else:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1).repeat(1, n_channels, 1, 1))
                print(new_proj.weight.shape)
            new_proj.bias = self.v.patch_embed.proj.bias
        self.mypatch_embed.proj = new_proj
        self.v.patch_embed = self.mypatch_embed
        #print(self.mypatch_embed.weight)
        #print(self.v.patch_embed.proj.weight)

        # if audio_set_pretrain: We also normalize the input audio spectrogram so that
        # the dataset mean and standard deviation are 0 and 0.5, respec
        # tively

        # Se la nuova t_dim (n_patch lungo asse temporale) è minore di quella originale, prende 
        # dei valori centrali da quella originale e li utilizza
        # Se invece è maggiore si fa interpolazione ovvero, se quella originale va da 5 a 16 e sono 4 valori
        # se si vuole farne 8 i valori centrali sono i valori centrali da 5 a 16.
        if imagenet_pretrain == True:
            # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            # cut (from middle) or interpolate the second dimension of the positional embedding
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
            # cut (from middle) or interpolate the first dimension of the positional embedding
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            # flatten the positional embedding
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
            # concatenate the above positional embedding with the cls token and distillation token of the deit model.
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        else:
            # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
            self.v.patch_embed.num_patches = int((input_fdim / fstride) * (input_tdim / tstride))
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02) # inizializza i pesi con valori estratti da una normale, si usa perchè aiuta le prime fasi di train
        

    def get_shape(self, fstride, tstride, input_fdim=32, input_tdim=1008):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim #a me 2, 63
    
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        if self.single == False:
            x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2 #perchè il distill ha due cls heads
        return x
    
class pret_MVIT(nn.Module):
    def __init__(self, n_channels=22, img_height=224, img_width = 224 ,patch_size=16,
                 embed_dim=768, num_classes=4, single = False):
        super().__init__()
        if single == False:
            print("You are using pret MVIT")
        else:
            print("You are using classic pret VIT")
        if single == False:
            self.encoders = nn.ModuleList([
                pret_VIT(label_dim=num_classes, fstride=patch_size, tstride=patch_size, 
                        input_fdim=img_height, input_tdim=img_width, imagenet_pretrain=True, 
                        audioset_pretrain=False, model_size='base384', verbose=True, single = False)
                for _ in range(n_channels)
            ])
        else:
            self.encoder = pret_VIT(label_dim=num_classes, fstride=patch_size, tstride=patch_size, 
                        input_fdim=img_height, input_tdim=img_width, imagenet_pretrain=True, 
                        audioset_pretrain=False, model_size='base384', verbose=True, single = True)
            
        # classifier per output concatenato
        self.concat_classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * n_channels),
            nn.Linear(embed_dim * n_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # classifier per output singolo
        self.single_classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.single = single

    def forward(self, x):
        # in questo modo devo dare in input tutti gli spettrogrammi concatenati sulla profondità
        # x: [B, C, H, W] = [B, 22, 32, 32]

        # MVIT
        if self.single == False:
            tokens = []
            for i, encoder in enumerate(self.encoders):
                channel_i = x[:, i:i+1, :, :]  # [B, 1, H, W]
                channel_i = channel_i.squeeze(1).permute(0, 2, 1)  # [B, W, H] per input al vit pretrained che vuole 1 canale
                token = encoder(channel_i)     # [B, D]
                tokens.append(token)
            concat_token = torch.cat(tokens, dim=-1)  # [B, 22*D]
            out = self.concat_classifier(concat_token)
        # SINGLE VIT
        else:
            x = x.permute(0,1, 3, 2)
            single_token = self.encoder(x)
            # print(single_token.shape)
            out = self.single_classifier(single_token)      # [B, num_classes]

        return out


# print(timm.__version__)
# PATCH_SIZE = 16
# SINGLE = True
# #print([m for m in timm.list_models() if 'distilled' in m])
# model = pret_MVIT(n_channels=22, img_height = 32, img_width = 1008, patch_size=PATCH_SIZE, embed_dim=768, num_classes=4, single=SINGLE)
# test_input = torch.rand([10,22,32, 1008])
# test_output = model(test_input)
# print(test_output.shape)

