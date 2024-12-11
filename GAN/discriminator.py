import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.flatten = nn.Flatten()
        self.leaky_relu = nn.LeakyReLU()
        self.linear = nn.Linear(128*4*4, 2)


    def forward(self, x):
        # input size [B, 3, 64, 64]
        x = self.conv1(x)   # output size [B, 16, 32, 32]
        x = self.leaky_relu(x)
        x = self.conv2(x)   # output size [B, 32, 16, 16]
        x = self.leaky_relu(x)
        x = self.conv3(x)   # output size [B, 64, 8, 8]
        x = self.leaky_relu(x)
        x = self.conv4(x)   # output size [B, 128, 4, 4]
        x = self.leaky_relu(x)
        flattened = self.flatten(x)
        x = self.linear(flattened)

        return x





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

    outputs = model(images)
    print(outputs.shape)



if __name__ == "__main__":
    main()