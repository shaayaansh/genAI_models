import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from discriminator import Discriminator
from generator import Generator

class GAN(nn.Module):
    def __init__(self, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.l_dim = latent_dim

    def forward(self, batch_size):
        """
        Forward pass for inference: generate fake images
        """
        noise = torch.randn(batch_size, self.l_dim)
        fake_image = self.generator(noise)
        
        return fake_image
