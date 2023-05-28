from swin_trm import PatchEmbedding
import torch


def patch_embedding_testing():
    x = torch.rand(2, 3, 224, 224)
    model = PatchEmbedding(in_channels=3, embedding_channels=768, patch_size=16)
    output = model(x)
    assert output.shape == torch.Size([2, 196, 768])


if __name__ == '__main__':
    patch_embedding_testing()
