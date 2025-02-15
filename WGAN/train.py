import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from generator import Generator
from critic import Critic
from wgan import WGAN
from data_loader import DataLoad
from torchvision.transforms.functional import to_pil_image
from pathlib import Path


def main():
    loader = DataLoad("train", batch_size=32)
    train_dataset, train_dataloader = loader.load()

    num_epochs = 300
    disc_lr = 1e-5
    gen_lr = 1e-5
    model = WGAN(latent_dim=100)

    model.train()
    generator = model.generator
    critic = model.critic
    critic_loss = nn.BCEWithLogitsLoss()
    gen_optimizer = AdamW(generator.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    disc_optimizer = AdamW(critic.parameters(), lr=disc_lr, betas=(0.5, 0.999))
    
    for epoch in range(num_epochs):
        epoch_loss_critic = 0
        epoch_accuracy_critic = 0
        total_samples = 0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            real_images, _ = batch
            batch_size = real_images.shape[0]
            total_samples += batch_size

            noise = torch.randn(batch_size, 100)
            fake_images = generator(noise)

            # wgan labels are 1 and -1
            real_labels = torch.ones(batch_size, 1).float()
            fake_labels = torch.full((batch_size, 1), -1).float()
            
            # Stack real and fake images
            all_images = torch.cat((real_images, fake_images), dim=0)  
            all_labels = torch.cat((real_labels, fake_labels), dim=0)
            permutation = torch.randperm(all_images.size(0))  # Generate random permutation indices
            shuffled_images = all_images[permutation]
            shuffled_labels = all_labels[permutation]
            # -------------------------
            # Train Critic
            # -------------------------
            disc_optimizer.zero_grad()
            scores = critic(shuffled_images)
            c_loss = - torch.mean(scores * shuffled_labels)
            c_gp = model.gradient_penalty(batch_size, real_images, fake_images)
            loss = c_loss + c_gp
            c_loss.backward()
            disc_optimizer.step()  
            # -------------------------
            # Train Generator
            # -------------------------
            if idx % 5 == 0:
                gen_optimizer.zero_grad()
                noise = torch.randn(batch_size, 100).to(real_images.device)
                fake_images = generator(noise)
                fake_scores = critic(fake_images)
                gen_loss = - torch.mean(fake_scores * real_labels)
                gen_loss.backward()
                gen_optimizer.step()


        gen_image(generator, epoch)


def gen_image(generator, epoch):
    generator.eval()  

    batch_size = 1  
    latent_dim = 100  
    noise = torch.randn(batch_size, latent_dim)  

    with torch.no_grad(): 
        fake_images = generator(noise)  
        fake_images = (fake_images + 1) / 2  

    save_dir = Path("generated_images")
    save_dir.mkdir(parents=True, exist_ok=True)

    
    for i, img_tensor in enumerate(fake_images):
        img = to_pil_image(img_tensor.squeeze(0)) 
        img.save(save_dir / f"EPOCH_{epoch}-gen_image_{i + 1}.png")



if __name__ == "__main__":
    main()