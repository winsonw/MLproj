from simpleCNN import SimpleCNN
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
