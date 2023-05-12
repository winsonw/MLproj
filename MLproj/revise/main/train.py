import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from simpleCNN import SimpleCNN
from data import load_data
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Trainer:
    def __init__(self, model, parameters, train_loader, val_loader):
        self.model = model
        self.parameters = parameters
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = self.parameters["device"]
        self.num_epoch = self.parameters["num_epoch"]
        self.lr = self.parameters["lr"]
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.losses = [[],[]]

    def save_model(self, path):
        torch.save(self.model.state_dict(), path + "/simpleCNN.pt")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path + "/simpleCNN.pt"))

    def plot(self):
        plt.plot(self.losses[0])
        plt.plot(self.losses[1])
        plt.show()

    def train(self):
        num_train, num_val = len(self.train_loader), len(self.val_loader)
        for epoch in range(self.num_epoch):
            train_loss = 0.0
            for i, (images, label) in enumerate(self.train_loader):
                images = images.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()
                predict = self.model(images)
                loss = self.loss_function(predict, label)
                train_loss += loss
                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{self.num_epoch}], Step [{i + 1}], Loss: {loss.item():.4f}")

                loss.backward()
                self.optimizer.step()
            self.losses[0].append(train_loss/num_train)
            self.losses[1].append(self.evaluate(self.val_loader) / num_val)

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


def main(train=True, model_class=SimpleCNN):
    model = model_class(num_class=10)
    parameters = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "num_epoch": 2,
        "lr": 0.001,
    }

    train_loader, val_loader, test_loader = load_data("../data")
    trainer = Trainer(model, parameters, train_loader, val_loader)

    if train:
        trainer.train()
        trainer.save_model("../model_para")
        trainer.plot()
    else:
        trainer.load_model("../model_para")

    print(trainer.evaluate(test_loader) / len(test_loader))


if __name__ == '__main__':
    main()
    # SimpleCNN(False)
