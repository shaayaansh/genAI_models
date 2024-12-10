import torch
import torch.nn as nn
from torch.optim import AdamW
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score



class MyConvModel(nn.Module):
    def __init__(self):
        super(MyConvModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32*32*32, 10)
        self.relu = nn.ReLU()


    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        flatten = self.flatten(conv2)
        out = self.linear(flatten)

        return out

def train(model, num_epochs, train_dataloader, test_dataloader, lr=1e-4):
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch} Started \n")
        
        epoch_loss = 0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            
            optimizer.zero_grad()
            inputs, labels = batch[0], batch[1]
            output = model(inputs)
            loss = loss_fn(output, labels)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        print(f"Epoch Loss: {epoch_loss}")
        print("==============================")

        if epoch % 2 == 0:
            print(f"Evaluating Model in Epoch: {epoch}")
            print(f"Test Accuracy Score: {test(model, test_dataloader)}")
            print("===================================")

def test(model, test_dataloader):
    model.eval()
    y_preds = []
    y_true = []
    for idx, batch in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            inputs, labels = batch[0], batch[1]
            outputs = model(inputs)
            y_pred = torch.argmax(outputs, dim=-1)
            y_preds.extend(y_pred.numpy())
            y_true.extend(labels.numpy())
    
    return accuracy_score(y_true, y_preds)


def main():
    model = MyConvModel()

    transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]
)

    # Download CIFAR10 train dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root = "data",
        train=True,
        download=True,
        transform=transform
    )

    # Download CIFAR10 test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root = "data",
        train=False,
        download=True,
        transform=transform
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True
    )

    train(model, 10, train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()




