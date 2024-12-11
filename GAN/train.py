import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator


def main():
    transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FakeData(
        size=1000,
        transform=transform,
        target_transform=None,
        random_offset=0
    )

    test_dataset = torchvision.datasets.FakeData(
        size=200,
        transform=transform,
        target_transform=None,
        random_offset=1000
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True
    )
    model = Discriminator()
    images, labels = next(iter(train_dataloader))

    print(images[0])
    outputs = model(images)
    print(outputs.shape)

    generator = Generator(100, 64)
    print(generator(images))

if __name__ == "__main__":
    main()