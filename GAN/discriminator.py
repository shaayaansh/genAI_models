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
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(128*4*4, 2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # input size [B, 3, 64, 64]
        x = self.conv1(x)   # output size [B, 16, 32, 32]
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv2(x)   # output size [B, 32, 16, 16]
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv3(x)   # output size [B, 64, 8, 8]
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv4(x)   # output size [B, 128, 4, 4]
        x = self.sigmoid(x)
        flattened = self.flatten(x)
        x = self.linear(flattened)
        x = self.sigmoid(x) # output size [B, 2]

        return x
