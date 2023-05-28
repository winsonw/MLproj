import torch.nn as nn
import torch


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_channels, patch_size, norm_layer=nn.GELU):
        super(PatchEmbedding, self).__init__()

        self.embedding_layer = nn.Conv2d(in_channels, embedding_channels, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer()

    def forward(self, x):
        # x:[B, C, H, W]
        x = self.embedding_layer(x)
        B, C, _, _ = x.shape
        x = x.view([B, C, -1]).transpose(1, 2)
        return self.norm(x)


