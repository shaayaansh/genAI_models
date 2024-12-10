import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
import os
from vae import visualize_and_save


def main():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.CelebA(
        root="./celeb_data",
        split='train',
        target_type='attr',  
        transform=transform,
        download=True
    )

    val_dataset = datasets.CelebA(
        root="./celeb_data",
        split='valid',
        target_type='attr',
        transform=transform,
        download=True
    )

    train_dataloadet = DataLoader(
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
    optimizer = AdamW(model.parameters(), lr=0.0001)
    model.train()
    num_epochs = 15

    save_directory = 'celeb_vae_reconstructions'
    os.makedirs(save_directory, exist_ok=True)

    for epoch in range(num_epochs):
        for idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            images, labels = batch
            outputs, z_mean, z_log_var = model(images)
            reconstruction_loss = loss_fn(outputs, images)
            kl_loss = -0.5 * torch.sum(1 + 2 * z_log_var - z_mean**2 - torch.exp(2 * z_log_var))
            total_loss = reconstruction_loss + 0.005 * kl_loss
            total_loss.backward()
            optimizer.step()

        visualize_and_save(model, test_dataloader, epoch, save_dir=save_directory)





class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,          # Changed from 1 to 3 for RGB
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1               # Added padding to control spatial dimensions
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # Updated input features from 3*3*128 to 7*7*128 = 6272
        self.linear_mean = nn.Linear(7*7*128, 10)
        self.linear_log_var = nn.Linear(7*7*128, 10)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))     # Output: [batch, 32, 32, 32]
        conv2 = self.relu(self.conv2(conv1)) # Output: [batch, 64, 16, 16]
        conv3 = self.relu(self.conv3(conv2)) # Output: [batch, 128, 8, 8]
        flattened = self.flatten(conv3)       # Output: [batch, 128*8*8=8192]
        z_mean = self.linear_mean(flattened)  # Output: [batch, 10]
        z_log_var = self.linear_log_var(flattened)  # Output: [batch, 10]
        
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(10, 128*8*8)  # Updated to match Encoder's output
        self.relu = nn.ReLU()
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
            out_channels=3,          # Changed from 1 to 3 for RGB
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.linear(z)                    # Output: [batch, 128*8*8]
        z = z.view(z.shape[0], 128, 8, 8)    # Reshape to [batch, 128, 8, 8]
        x = self.relu(self.conv1_transpose(z))  # Output: [batch, 64, 16, 16]
        x = self.relu(self.conv2_transpose(x))  # Output: [batch, 32, 32, 32]
        x = self.relu(self.conv3_transpose(x))  # Output: [batch, 16, 64, 64]
        x = self.sigmoid(self.conv(x))          # Output: [batch, 3, 64, 64]
        
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


if __name__ == "__main__":
    main()