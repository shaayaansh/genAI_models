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
from torchvision.transforms.functional import to_pil_image
from pathlib import Path


def main():
    loader = DataLoad("train")
    train_dataset, train_dataloader = loader.load()

    num_epochs = 50
    disc_lr = 1e-5
    gen_lr = 1e-5
    model = GAN(latent_dim=100)

    model.train()
    generator = model.generator
    discriminator = model.discriminator
    discriminator_loss = nn.BCEWithLogitsLoss()
    gen_optimizer = AdamW(generator.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    disc_optimizer = AdamW(discriminator.parameters(), lr=disc_lr, betas=(0.5, 0.999))
    
    for epoch in range(num_epochs):
        epoch_loss_disc = 0
        epoch_accuracy_disc = 0
        total_samples = 0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            real_images, _ = batch
            batch_size = real_images.shape[0]
            total_samples += batch_size

            noise = torch.randn(batch_size, 100)
            fake_images = generator(noise)

            # smooth the labels
            smooth_real_labels = torch.full((batch_size, 1), 0.9).float()
            real_labels = torch.ones(batch_size, 1).float()
            fake_labels = torch.zeros(batch_size, 1).float()
            
            # Stack real and fake images
            all_images = torch.cat((real_images, fake_images), dim=0)  
            all_labels = torch.cat((smooth_real_labels, fake_labels), dim=0)
            permutation = torch.randperm(all_images.size(0))  # Generate random permutation indices
            shuffled_images = all_images[permutation]
            shuffled_labels = all_labels[permutation]
            # -------------------------
            # Train Discriminator
            # -------------------------
            disc_optimizer.zero_grad()

            disc_output = discriminator(shuffled_images)
            disc_loss = discriminator_loss(disc_output, shuffled_labels)
            epoch_loss_disc += disc_loss
            disc_loss.backward()
            disc_optimizer.step()  
            # Calculate accuracy
            predictions = (disc_output > 0.5).float()
            binary_labels = (shuffled_labels > 0.5).float()
            correct_predictions = (predictions == binary_labels).sum().item()

            epoch_accuracy_disc += correct_predictions
            # -------------------------
            # Train Generator
            # -------------------------
            gen_optimizer.zero_grad()
            noise = torch.randn(batch_size, 100).to(real_images.device)
            fake_images = generator(noise)
            disc_output_fake = discriminator(fake_images)
            gen_loss = discriminator_loss(disc_output_fake, real_labels)
            gen_loss.backward()
            gen_optimizer.step()

        epoch_accuracy_disc /= (total_samples * 2)
        print(f"EPOCH {epoch} LOSS OF DISCRIMINATOR: {epoch_loss_disc / len(train_dataloader)}")
        print(f"EPOCH {epoch} Discriminator Accuracy: {epoch_accuracy_disc:.4f}")
        gen_image(generator, epoch)


def gen_image(generator, epoch):
    generator.eval()  

    batch_size = 4  
    latent_dim = 100  
    noise = torch.randn(batch_size, latent_dim)  

    with torch.no_grad(): 
        fake_images = generator(noise)  
        fake_images = (fake_images + 1) / 2  

    fig, axes = plt.subplots(4, 4, figsize=(32, 32)) 

    save_dir = Path("generated_images")
    save_dir.mkdir(parents=True, exist_ok=True)

    
    for i, img_tensor in enumerate(fake_images):
        img = to_pil_image(img_tensor.squeeze(0)) 
        img.save(save_dir / f"EPOCH_{epoch}-gen_image_{i + 1}.png")



if __name__ == "__main__":
    main()