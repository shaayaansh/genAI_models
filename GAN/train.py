import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
from gan import GAN
from data_loader import DataLoad

def main():

    loader = DataLoad("train")
    train_dataset, train_dataloader = loader.load()

    model = Discriminator()
    images, labels = next(iter(train_dataloader))

    print(images.shape)
    outputs = model(images)
    print(outputs.shape)

if __name__ == "__main__":
    main()