import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
import os
import numpy as np


def main():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    ])

    data_root = "./data/celeba"
    dataset = datasets.ImageFolder(root=data_root, transform=transform)

    train_ratio = 0.8
    val_ratio = 0.2
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=True
    )

    model = VAE()
    model.train()

    loss_fn = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=0.001)
    model.train()
    num_epochs = 10

    save_directory = 'celeb_vae_reconstructions'
    os.makedirs(save_directory, exist_ok=True)

    for epoch in range(num_epochs):
        for idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            images, labels = batch
            outputs, z_mean, z_log_var = model(images)
            reconstruction_loss = loss_fn(outputs, images)
            kl_loss = -0.5 * torch.sum(1 + 2 * z_log_var - z_mean**2 - torch.exp(2 * z_log_var))
            total_loss = reconstruction_loss + 0.05 * kl_loss
            total_loss.backward()
            optimizer.step()

        visualize_and_save(model, val_dataloader, epoch, save_dir=save_directory)





class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,          # Changed from 1 to 3 for RGB
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1               # Added padding to control spatial dimensions
        )
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.linear_mean = nn.Linear(8*8*128, 200)
        self.linear_log_var = nn.Linear(8*8*128, 200)

    def forward(self, x):
        conv1 = self.leaky_relu(self.conv1(x))  
        conv1 = self.batch_norm1(conv1)   
        conv2 = self.leaky_relu(self.conv2(conv1)) 
        conv2 = self.batch_norm2(conv2)
        conv3 = self.leaky_relu(self.conv3(conv2)) 
        conv3 = self.batch_norm3(conv3)
        flattened = self.flatten(conv3)  
        z_mean = self.linear_mean(flattened)  
        z_log_var = self.linear_log_var(flattened) 
        
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(200, 128*8*8) 
        self.leaky_relu = nn.LeakyReLU()
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.batch_norm4 = nn.BatchNorm2d(3)
        self.conv1_transpose = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.conv2_transpose = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.conv3_transpose = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.conv = nn.Conv2d(
            in_channels=16,
            out_channels=3,     
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.linear(z)                    
        z = z.view(z.shape[0], 128, 8, 8)    
        x = self.leaky_relu(self.conv1_transpose(z))  
        x = self.batch_norm1(x)
        x = self.leaky_relu(self.conv2_transpose(x)) 
        x = self.batch_norm2(x) 
        x = self.leaky_relu(self.conv3_transpose(x)) 
        x = self.batch_norm3(x)
        x = self.batch_norm4(self.conv(x))
        x = self.sigmoid(x)          
        
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * log_var) * epsilon
        return z

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(z)
        return decoded, z_mean, z_log_var



def visualize_and_save(model, dataloader, epoch, save_dir='celeb_vae_reconstructions'):
    """
    Visualizes and saves original and reconstructed images from the VAE.

    Args:
        model (nn.Module): The trained VAE model.
        dataloader (DataLoader): DataLoader for the dataset to visualize.
        epoch (int): Current epoch number.
        save_dir (str): Directory to save the reconstruction images.
    """
    model.eval()  
    with torch.no_grad():
        test_image, test_label = next(iter(dataloader))
        decoded, z_mean, z_log_var = model(test_image)
    
    num_images = 8  
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i in range(num_images):
        
        original_img = test_image[i].cpu().numpy()
        original_img = (original_img * 0.5) + 0.5  
        original_img = np.transpose(original_img, (1, 2, 0))
        original_img = np.clip(original_img, 0, 1)  
        
        gen_img = decoded[i].cpu().numpy()
        gen_img = (gen_img * 0.5) + 0.5 
        gen_img = np.transpose(gen_img, (1, 2, 0)) 
        gen_img = np.clip(gen_img, 0, 1)  
        
        axs[0, i].imshow(original_img)
        axs[0, i].set_title("Original")
        axs[0, i].axis('off')

        axs[1, i].imshow(gen_img)
        axs[1, i].set_title("Reconstructed")
        axs[1, i].axis('off')
    
    plt.suptitle(f'Epoch {epoch + 1}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    os.makedirs(save_dir, exist_ok=True)
    
    filename = os.path.join(save_dir, f'epoch_{epoch + 1}.jpg')
    plt.savefig(filename, format='jpg')
    plt.close(fig) 
    model.train() 


if __name__ == "__main__":
    main()