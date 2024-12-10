import torch
import torch.nn as nn
from torch.optim import AdamW
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class MlpModel(nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        flatten = self.flatten(x)
        h1 = self.relu(self.fc1(flatten))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))

        return h3


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

        if epoch % 10 == 0:
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
    model = MlpModel()

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

    train(model, 15, train_dataloader, test_dataloader)

if __name__ == "__main__":
    main()
    



