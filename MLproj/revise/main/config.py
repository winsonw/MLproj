from simpleCNN import SimpleCNN
from unet import Unet
from resnet import ResNet, BottleneckBlock, ResidualBock
from vit import VisionTransformer
import torch
import torch.nn as nn
import torch.optim as optim


class Config:
    @staticmethod
    def get_train_config(neural_network):
        if neural_network == "SimpleCNN":
            parameter = {
                "model": SimpleCNN,
                "name_model": "SimpleCNN",
                "num_class": 10,
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "num_epoch": 2,
                "lr": 0.001,
                "loss_function": nn.CrossEntropyLoss,
                "optimizer": optim.Adam,
            }
            return parameter

        if neural_network == "Unet":
            parameter = {
                "model": Unet,
                "name_model": "Unet",
                "num_class": 10,
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "num_epoch": 2,
                "lr": 0.001,
                "loss_function": nn.CrossEntropyLoss,
                "optimizer": optim.Adam,
            }
            return parameter

        if neural_network == "ResNet":
            parameter = {
                "model": ResNet,
                "name_model": "ResNet",
                "res_block": ResidualBlock,
                "num_class": 10,
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "num_epoch": 2,
                "lr": 0.001,
                "loss_function": nn.CrossEntropyLoss,
                "optimizer": optim.Adam,
                "layers": [2, 2, 2, 2],
                "d_ff": 500,
            }
            return parameter

        if neural_network == "ResNet50":
            parameter = {
                "model": ResNet,
                "name_model": "ResNet50",
                "res_block": BottleneckBlock,
                "num_class": 10,
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "num_epoch": 2,
                "lr": 0.001,
                "loss_function": nn.CrossEntropyLoss,
                "optimizer": optim.Adam,
                "d_ff": 1000,
                "bottleneck_channels": [64, 128, 256, 512],
                "layers": [3, 4, 6, 3]
            }
            return parameter

        if neural_network == "ViT":
            parameter = {
                "model": VisionTransformer,
                "name_model": "ViT",
                "num_class": 10,
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "num_epoch": 10,
                "lr": 0.001,
                "loss_function": nn.CrossEntropyLoss,
                "optimizer": optim.Adam,
            }
            return parameter
