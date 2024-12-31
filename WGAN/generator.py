import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, hidden_dim):
        super(Generator, self).__init__()
        self.conv1_transpose = nn.ConvTranspose2d(
            in_channels=100,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.conv2_transpose = nn.ConvTranspose2d(
            in_channels=512,        
            out_channels=256,       
            kernel_size=3,       
            stride=2,             
            padding=1,            
            output_padding=1   
        )
        self.conv3_transpose = nn.ConvTranspose2d(
            in_channels=256,        
            out_channels=128,       
            kernel_size=3,       
            stride=2,             
            padding=1,            
            output_padding=1      
        )
        self.conv4_transpose = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.conv5_transpose = nn.ConvTranspose2d(
            in_channels=64,       
            out_channels=1,        
            kernel_size=3,         
            stride=2,              
            padding=3,
            output_padding=1   
        )
        self.hidden_dim = hidden_dim
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.batch_norm1 = nn.BatchNorm2d(512)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(64)
    

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.hidden_dim, 1, 1)    # size: [B, hidden_dim, 1, 1]
        x = self.conv1_transpose(x)                     # size: [B, 512, 2, 2]
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.conv2_transpose(x)                     # size: [B, 256, 4, 4]
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.conv3_transpose(x)                     # size: [B, 128, 8, 8]
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)
        x = self.conv4_transpose(x)                     # size: [B, 64, 16, 16]
        x = self.batch_norm4(x)
        x = self.leaky_relu(x)
        x = self.conv5_transpose(x)                     # size: [B, 1, 28, 28]
        x = self.tanh(x)

        return x
