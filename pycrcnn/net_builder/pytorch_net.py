# Example of image classifier example for Fashion-MNIST dataset, from
# Deeplizard.

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


train_set = torchvision.datasets.MNIST(
    root = './data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=100,
    shuffle=True
)

net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=4*4*32, out_features=256),
    nn.Linear(in_features=256, out_features=10)
)

# Helper functions


def show_img(img):
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def new_set():
    return torchvision.datasets.FashionMNIST(
                                root='./data'
                                , train=True
                                , download=True
                                , transform=transforms.Compose([
                                    transforms.ToTensor()
                                ])
    )


def new_batch(size):
    return torch.utils.data.DataLoader(
                                train_set
                                , batch_size=size
                                , shuffle=False
    )


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def train_net(network, epochs):

    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for i in range(0, epochs):
        total_loss = 0
        total_correct = 0

        for batch in train_loader:
            images, labels = batch

            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()  # Resets gradients
            loss.backward()  # Calc gradients
            optimizer.step()  # Update weights

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        print(
            "epoch: ", i+1,
            "\ntotal_correct: ", total_correct,
            "\nloss: ", total_loss)


@torch.no_grad()
def use_net(net, batch):
    return net(batch)


def save_net(path="./model.pt"):
    torch.save(net.state_dict(), path)


def load_net(path="./model.pt"):
    net.load_state_dict(torch.load(path))
    net.eval()

    return net


#train_net(net, 2)
#save_net()
