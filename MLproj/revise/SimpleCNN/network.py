from abc import ABC

import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_class):
        super(SimpleCNN, self).__init__()
        self.conv1_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*14*14, 128),
            nn.ReLU(),
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        x = self.conv1_layers(x)
        x = self.fc_layers(x)
        return x
