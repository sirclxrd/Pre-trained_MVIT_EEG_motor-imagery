import torch
import torch.nn as nn
import timm

class CustomViT22(nn.Module):
    def __init__(self, num_channels=22, num_classes=4, img_size=32):
        super().__init__()
        # Carica ViT base con patch 16 e immagini 32x32
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Cambia input layer per accettare 22 canali invece di 3
        old_conv = self.vit.patch_embed.proj
        self.vit.patch_embed.proj = nn.Conv2d(
            in_channels=num_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )

        # Cambia l'head finale per 4 classi
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

# Dummy input: [batch, 22, 32, 32]
model = CustomViT22()
dummy_input = torch.randn(8, 22, 224, 224)
output = model(dummy_input)
print(output.shape)  # --> torch.Size([8, 4])
    
# print(timm.list_models('vit*'))  # stampa tutti i ViT disponibili

