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


    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = torch.randn(batch_size, 1, 1, 1)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        interpolated.requires_grad_()

        preds = self.critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=preds,
            inputs=interpolated,
            grad_outputs=torch.ones_like(preds),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        grad_norms = torch.sqrt(torch.sum(gradients ** 2, dim=(1,2,3)))
        gradient_penalty = torch.mean((grad_norms - 1.0) ** 2)

        return gradient_penalty
