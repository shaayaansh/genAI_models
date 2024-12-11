import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
from gan import GAN
from data_loader import DataLoad


def main():
    loader = DataLoad("train")
    train_dataset, train_dataloader = loader.load()

    num_epochs = 5
    lr = 0.0001
    model = GAN(latent_dim=100)

    model.train()
    generator = model.generator
    discriminator = model.discriminator
    discriminator_loss = nn.CrossEntropyLoss()
    gen_optimizer = AdamW(generator.parameters(), lr=lr)
    disc_optimizer = AdamW(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for idx, batch in enumerate(tqdm(train_dataloader)):
            real_images, _ = batch
            batch_size = real_images.shape[0]
            noise = torch.randn(batch_size, 100)
            fake_images = generator(noise, batch_size)
            
            real_labels = torch.ones(batch_size, 1).long().squeeze(1)
            fake_labels = torch.zeros(batch_size, 1).long().squeeze(1)
            
            disc_optimizer.zero_grad()
            disc_output_real = discriminator(real_images)
            disc_output_fake = discriminator(fake_images.detach())
            disc_loss_real = discriminator_loss(disc_output_real, real_labels)
            disc_loss_fake = discriminator_loss(disc_output_fake, fake_labels)
            disc_loss = disc_loss_real + disc_loss_fake
            disc_loss.backward()
            disc_optimizer.step()   

            gen_optimizer.zero_grad()
            disc_output_fake = discriminator(fake_images)
            gen_loss = discriminator_loss(disc_output_fake, real_labels)
            gen_loss.backward()
            gen_optimizer.step()



if __name__ == "__main__":
    main()