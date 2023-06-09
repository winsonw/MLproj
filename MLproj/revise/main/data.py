from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import torch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def load_test_data(root="../data", batch_size=64, name_dataset="CIFAR10"):
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def load_all_data(root="../data", batch_size=64, name_dataset="CIFAR10"):
    full_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, load_test_data(root, batch_size, name_dataset)
