import torch
import torch.nn as nn
from attention import MultiHeadAttention


class EncodeBlock(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super(EncodeBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.att_layer = MultiHeadAttention(d_model, d_model//h, d_model//h, h, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ff_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        att = self.norm1(x)
        att = self.att_layer(att, att, att)
        x = x + att
        x = self.norm2(x)

        ff = self.ff_layer(x)
        x = x + ff
        return x


class VisionTransformer(nn.Module):
    def __init__(self, d_model, d_ff, h,  image_size, patch_size, num_layers, trm=EncodeBlock, in_channel=3, dropout=0.1):
        super(VisionTransformer, self).__init__()
        num_patch = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(in_channel, d_model, stride=patch_size, kernel_size=patch_size)
        self.pos_encoding = nn.Parameter(torch.randn((1, num_patch+1, d_model)))
        self.class_token = nn.Parameter(torch.randn((1, 1, d_model)))
        self.transformer_blocks = nn.ModuleList([trm(d_model, d_ff, h, dropout) for _ in range(num_layers)])

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(2, 1)  # [batch_size, num_patch, d_model]
        class_token = self.class_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([class_token, x], dim=1)

        x = x + self.pos_encoding

        for layer in self.transformer_blocks:
            x = layer(x)

        return x


if __name__ == '__main__':
    # Assume the input images have size 224x224 with 3 channels (RGB), and we are using 8 heads, 512-dimensional embeddings,
    # and 6 layers in the transformer. The image is divided into patches of size 16x16.
    model = VisionTransformer(d_model=512, d_ff=2048, h=8, image_size=224, patch_size=16, num_layers=6)

    # Assume we have a batch of 10 images
    input_images = torch.randn(10, 3, 224, 224)

    # Forward pass
    output = model(input_images)

    # The output should have shape [batch_size, num_patches + 1 (for the class token), d_model]
    print(output.shape)