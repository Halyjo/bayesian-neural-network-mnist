from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from imageio import imread
from torch.utils.data.dataset import Dataset
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import sys


def accuracy(pred, lab):
    correct = (pred == lab).sum()
    total = torch.tensor(pred.shape).prod()
    return (correct/total).item()


## Model
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 4*4 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act = F.relu
        self.softmax = nn.Softmax(dim=1)
    
    def load(self, state_dict_path:str):
        self.load_state_dict(torch.load(state_dict_path))


    def forward(self, x, apply_softmax=False):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(self.act(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(self.act(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        if apply_softmax:
            return self.softmax(x)
        return x


## Data Loaders
def get_data_loaders(batch_size=128):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "mnist-data/",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "mnist-data/",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, test_loader


def train(net, optimizer, critic, epochs=2):
    ## get data
    train_loader, test_loader = get_data_loaders()

    loss_vec = np.zeros(int(epochs * len(train_loader)))
    acc_vec = np.zeros(int(epochs * len(train_loader)))


    ## forall epochs
    for i in tqdm(range(epochs)):
        for j, (img_b, lab_b) in enumerate(train_loader):
            optimizer.zero_grad()

            ## Predict
            pred_b = net(img_b, apply_softmax=True)

            ## Compute loss
            loss = critic(pred_b, lab_b)
            
            ## Propagate gradients
            loss.backward()

            ## Update weights
            optimizer.step()

            ## Store loss for monitoring
            loss_vec[j] = loss.item()
            
            ## Measure prediction accuracy
            pred_cls = torch.argmax(pred_b, dim=1)
            acc_vec[j] = accuracy_score(lab_b, pred_cls)

            ## Print status
            sys.stdout.write(str(i+1) + '%\r')
            sys.stdout.flush()

            msg = f"Epoch: {i:<10}Loss: {loss_vec[j]:<10.04}Accuracy: {acc_vec[j]:<10}."
            sys.stdout.write(msg)
            sys.stdout.flush()
        
    ## Save model
    torch.save(net.state_dict(), "lenet_state_dict.pt")

    plt.plot(np.log(loss_vec))
    plt.show()
    plt.plot(acc_vec)
    plt.show()
    return loss, acc_vec


if __name__ == "__main__":
    batch_size = 128
    lr = 0.01

    ## Initialize model
    net = LeNet()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    critic = nn.CrossEntropyLoss()

    ## Train model
    loss, acc = train(net, optimizer, critic, epochs=1)
    