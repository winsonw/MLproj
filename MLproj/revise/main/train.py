import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import data
from config import Config
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Trainer:
    def __init__(self, arg, model=None):
        # self.parameters =arg
        # lr = arg["lr"]
        # num_class = arg["num_class"]
        self.name_model = arg["name_model"]
        self.device = arg["device"]
        if model is None:
            self.model = arg["model"](arg)
        else:
            self.model = model
        self.model.to(self.device)
        self.num_epoch = arg["num_epoch"]
        self.loss_function = arg["loss_function"]()
        self.optimizer = arg["optimizer"](self.model.parameters(), lr=arg["lr"])

        self.train_losses, self.val_losses = [], []

    def train(self, train_loader, val_loader):
        num_train, num_val = len(train_loader), len(val_loader)
        for epoch in range(self.num_epoch):
            train_loss = 0.0
            for i, (images, label) in enumerate(train_loader):
                images = images.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()
                predict = self.model(images)
                loss = self.loss_function(predict, label)
                train_loss += loss
                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{self.num_epoch}], Step [{i + 1}/{num_train}], Loss: {loss.item():.4f}")

                loss.backward()
                self.optimizer.step()
            self.train_losses.append(train_loss/num_train)
            self.val_losses.append(self.evaluate(val_loader) / num_val)

    def evaluate(self, dataloader):
        self.model.eval()
        losses = 0.0

        with torch.no_grad():
            for images, label in dataloader:
                images = images.to(self.device)
                label = label.to(self.device)

                predict = self.model(images)
                loss = self.loss_function(predict, label)
                losses += loss

        return losses

    def save_model(self, path):
        torch.save(self.model.state_dict(), path + self.name_model + ".pt")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path + self.name_model + ".pt"))

    def plot(self):
        plt.plot(self.train_losses, label="train_losses")
        plt.plot(self.val_losses, label="val_losses")
        plt.show()


def main(train=True, name_model="SimpleCNN"):
    data_path = "../data"
    model_path = "../model_para"
    name_dataset = "CIFAR10"
    # name_model = "SimpleCNN"
    # name_model = "Unet"
    # name_model = "ResNet"
    # name_model = "ViT"
    name_model = "ResNet50"
    batch_size = 64
    train = True

    parameters = Config.get_train_config(name_model)

    train_loader, val_loader, test_loader = data.load_all_data(data_path, batch_size, name_dataset)
    if name_model == "ViT":
        model = parameters["model"](d_model=256, d_ff=1024, h=8, image_size=32,
                                    patch_size=4, num_layers=6, num_classes=10)
        trainer = Trainer(parameters, model)
    else:
        trainer = Trainer(parameters)

    if train:
        trainer.train(train_loader, val_loader)
        trainer.save_model(model_path)
        trainer.plot()
    else:
        trainer.load_model(model_path)

    print(trainer.evaluate(test_loader) / len(test_loader))


if __name__ == '__main__':
    main()
