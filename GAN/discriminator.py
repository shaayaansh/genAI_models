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
            in_channels=1,
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
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(128*2*2, 1)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)


    def forward(self, x):
        # input size [B, 1, 28, 28]
        x = self.conv1(x)   # output size [B, 16, 15, 15]
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv2(x)   # output size [B, 32, 8, 8]
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv3(x)   # output size [B, 64, 4, 4]
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv4(x)   # output size [B, 128, 2, 2]
        x = self.batch_norm3(x)
        x = self.sigmoid(x)
        flattened = self.flatten(x)
        x = self.linear(flattened)
        x = self.sigmoid(x) # output size [B, 2]

        return x
