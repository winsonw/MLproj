from swin_trm import PatchEmbedding, PatchMerging
import torch


def patch_embedding_test():
    x = torch.rand(2, 3, 224, 224)
    model = PatchEmbedding(in_channels=3, embedding_channels=768, patch_size=16)
    output = model(x)
    assert output.shape == torch.Size([2, 196, 768])

def patch_merging_test():
    x = torch.rand(2, 196, 768)
    model = PatchMerging(input_resolution=(14, 14), in_channels=768)
    output = model(x)
    assert output.shape == torch.Size([2, 49, 1536])


if __name__ == '__main__':
    # patch_embedding_testing()
    patch_merging_test()