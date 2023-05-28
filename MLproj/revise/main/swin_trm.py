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


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, in_channels, norm_layer=nn.GELU):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.norm = norm_layer()
        self.linear = nn.Linear(in_channels*4, in_channels*2)

    def forward(self, x):
        B, L, C = x.shape
        x = x.view([B, self.input_resolution[0], self.input_resolution[1], C])
        x = torch.cat([
            x[:, ::2, ::2, :],
            x[:, 1::2, ::2, :],
            x[:, ::2, 1::2, :],
            x[:, 1::2, 1::2, :],
        ], dim=3)
        x = x.view([B, L // 4, 4*C])
        x = self.norm(x)
        x = self.linear(x)
        return x