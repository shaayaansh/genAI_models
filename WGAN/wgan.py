import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from critic import Critic
from generator import Generator

class WGAN(nn.Module):
    def __init__(self, latent_dim):
        super(WGAN, self).__init__()
        self.l_dim = latent_dim
        self.critic = Critic()
        self.generator = Generator(self.l_dim)
        
    def forward(self, batch_size):
        """
        Forward pass for inference: generate fake images
        """
        noise = torch.randn(batch_size, self.l_dim)
        fake_image = self.generator(noise)
        
        return fake_image
