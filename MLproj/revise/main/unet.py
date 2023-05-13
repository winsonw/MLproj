import torch.nn as nn
import torch
import torchvision


class Unet(nn.Module):
    def __init__(self, num_class):
        super(Unet, self).__init__()
        self.conv1_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
        )
        self.conv2_layer = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.conv3_layer = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
        )
        self.con4_layer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.con5_layer = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.crop1 = torchvision.transforms.CenterCrop([8, 8])
        self.crop2 = torchvision.transforms.CenterCrop([12, 12])
        self.t_conv1_layer = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.t_conv2_layer = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*10*10, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        sample = []
        x = self.conv1_layer(x)
        sample.append(x)
        x = self.pool(x)
        x = self.conv2_layer(x)
        sample.append(x)
        x = self.pool(x)
        x = self.conv3_layer(x)
        x = self.t_conv1_layer(x)
        t = sample.pop()
        x = torch.cat([x, torchvision.transforms.functional.center_crop(t, [x.shape[2], x.shape[3]])], dim=1)
        x = self.con4_layer(x)
        x = self.t_conv2_layer(x)
        t = sample.pop()
        x = torch.cat([x, torchvision.transforms.functional.center_crop(t, [x.shape[2], x.shape[3]])], dim=1)
        x = self.con5_layer(x)
        x = self.fc_layer(x)
        return x
