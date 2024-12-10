import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True
)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=2,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3*3*128, 2)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        flattened = self.flatten(conv3)
        z = self.linear(flattened)

        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1_transpose = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2
        )

        self.conv2_transpose = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2
        )

        self.conv3_transpose = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            output_padding=1
        )

        self.conv = nn.Conv2d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu = nn.ReLU()
        self.linear = nn.Linear(2, 3*3*128)
        self.sigmoid = nn.Sigmoid()


    def forward(self, z):
        z = self.linear(z)
        z = z.view(z.shape[0], 128, 3, 3)
        x = self.relu(self.conv1_transpose(z))
        x = self.relu(self.conv2_transpose(x))
        x = self.relu(self.conv3_transpose(x))
        x = self.sigmoid(self.conv(x))
        
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z)
        
        return decoded



def main():
    # initialize the model
    model = AutoEncoder()

    loss_fn = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=0.0001)
    model.train()
    num_epochs = 10


    for epoch in range(num_epochs):
        for idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            images, labels = batch
            outputs = model(images)
            loss = loss_fn(outputs, images)
            loss.backward()
            optimizer.step()

    
    
    model.eval()

    test_image, test_label = next(iter(test_dataloader))
    generated = model(test_image)

    num_images = 8

    fig, axs = plt.subplots(2, num_images, figsize=(10, 4))

    for i in range(num_images):
        # Extract original and generated images
        original_img = test_image[i].detach().cpu().numpy()
        gen_img = generated[i].detach().cpu().numpy()

        # Undo normalization (assuming mean=0.5, std=0.5)
        original_img = original_img * 0.5 + 0.5
        gen_img = gen_img * 0.5 + 0.5

        # Remove the channel dimension
        original_img = original_img.squeeze()
        gen_img = gen_img.squeeze()

        axs[0, i].imshow(original_img, cmap='gray')
        axs[0, i].set_title("Original")
        axs[0, i].axis('off')

        axs[1, i].imshow(gen_img, cmap='gray')
        axs[1, i].set_title("Generated")
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

