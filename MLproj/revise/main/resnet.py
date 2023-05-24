import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.projection = nn.Identity()

        self.activation = nn.ReLU()

    def forward(self, x):
        residual, out = self.projection(x), self.layers(x)
        return self.activation(out + residual)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels=64, bottleneck_channel=64, stride=1):
        super(BottleneckBlock, self).__init__()
        out_channels = bottleneck_channel*4
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channel, bottleneck_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channel, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.projection = nn.Identity()

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.projection(x) + self.layers(x)
        return self.act(x)


class ResNet(nn.Module):
    def __init__(self, arg=None):
        super(ResNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block = arg["res_block"]
        if arg["name_model"] == "ResNet":
            self.layer1 = self.make_layer(64, 64, 1, arg["layers"][0])
            self.layer2 = self.make_layer(64, 128, 2, arg["layers"][1])
            self.layer3 = self.make_layer(128, 256, 2, arg["layers"][2])
            self.layer4 = self.make_layer(256, 512, 2, arg["layers"][3])
            last = 512
        else:
            self.layer1 = self.make_layer(64, 64, 1, arg["layers"][0], True)
            self.layer2 = self.make_layer(256, 128, 2, arg["layers"][1], True)
            self.layer3 = self.make_layer(512, 256, 2, arg["layers"][2], True)
            self.layer4 = self.make_layer(1024, 512, 2, arg["layers"][3], True)
            last = 512*4

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(last, arg["d_ff"]),
            nn.ReLU(),
            nn.Linear(arg["d_ff"], arg["num_class"]),
        )

    def make_layer(self, in_channels, out_channels, stride, num_layer, is_bottleneck=False):
        layers = []
        layers.append(self.block(in_channels, out_channels, stride))
        for _ in range(1, num_layer):
            layers.append(self.block(out_channels * (4 if is_bottleneck else 1), out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(x)
        return x