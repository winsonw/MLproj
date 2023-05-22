import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

        self.fc_q = nn.Linear(d_model, d_k)
        self.fc_k = nn.Linear(d_model, d_k)
        self.fc_v = nn.Linear(d_model, d_v)

    def forward(self, query, key, value):

        weight = torch.softmax(torch.matmul(self.fc_q(query), self.fc_k(key).transpose(-1, -2)) / np.sqrt(self.d_k), -1)
        att = torch.matmul(weight, self.fc_v(value))
        return att


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_v = d_v
        self.multi_head = nn.ModuleList([ScaledDotProductAttention(d_model, d_k, d_v) for _ in range(h)])
        self.fc_layer = nn.Linear(h * d_v, d_model)

    def forward(self, query, key, value):
        num_batch = query.shape[0]
        x = torch.cat([att_layer(query, key, value) for att_layer in self.multi_head])
        x = self.fc_layer(x.view(num_batch, -1, self.h * self.d_v))
        return x


if __name__ == '__main__':
    t = MultiHeadAttention(512, 512, 512, 8)
    input = torch.randn(64, 50,  512)
    output = t(input, input, input)
    print(output.shape)
