import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn


class DataLoad():
    def __init__(self, split, batch_size=64):
        self.train = True if split == "train" else False
        self.batch_size = batch_size
    
    def load(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  
        ])

        dataset = torchvision.datasets.MNIST(
            root='./data',
            train=self.train,
            transform=transform,
            download=True
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        return dataset, dataloader