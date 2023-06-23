import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import os

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    )

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = conv_block(in_channels, growth_rate)

    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.downsample(x)

def valid(net, device, testloader, criterion):
    correct = 0
    total = 0
    loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()

    print(f"Loss of the network on the 10000 test images: {loss / total}")
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
    return loss / total, 100 * correct / total


class SimpleDenseNet(nn.Module):
    def __init__(self, growth_rate=12, num_layers=3, num_classes=10):
        super(SimpleDenseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(3, stride=2, padding=1),
            DenseBlock(64, growth_rate, num_layers),
            TransitionLayer(64+growth_rate*num_layers, 128),
            DenseBlock(128, growth_rate, num_layers),
            TransitionLayer(128+growth_rate*num_layers, 64),
            DenseBlock(64, growth_rate, num_layers),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64+growth_rate*num_layers, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 32
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SimpleDenseNet().to(device)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    net.train()
    dev_acc = []
    dev_loss = []

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        loop = tqdm(iterable=trainloader)

        for i, data in enumerate(loop):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            loop.set_description(f'Epoch : {epoch+1}')
            loop.set_postfix(loss=running_loss / (i + 1))

        loss, acc = valid(net, device, testloader, criterion)

        dev_acc.append(acc)
        dev_loss.append(loss)

    print('Finished Training')

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, 21), dev_loss)
    plt.title('validation loss')
    plt.savefig('validation_loss.png')  # Add this line to save the figure

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, 21), dev_acc)
    plt.title('validation accuracy')
    plt.savefig('validation_accuracy.png')  # Add this line to save the figure

