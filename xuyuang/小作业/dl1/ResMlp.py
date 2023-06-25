#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  
from torch import nn
from torch.nn import functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# In[2]:


class AffineTransform(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x):
        return self.alpha * x + self.beta


class CommunicationLayer(nn.Module):
    def __init__(self, num_features, num_patches):
        super().__init__()
        self.aff1 = AffineTransform(num_features)
        self.fc1 = nn.Linear(num_patches, num_patches)
        self.aff2 = AffineTransform(num_features)

    def forward(self, x):
        x = self.aff1(x)
        residual = x
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        x = self.aff2(x)
        out = x + residual
        return out


class FeedForward(nn.Module):
    def __init__(self, num_features, expansion_factor):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.aff1 = AffineTransform(num_features)
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.aff2 = AffineTransform(num_features)

    def forward(self, x):
        x = self.aff1(x)
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.aff2(x)
        out = x + residual
        return out


class ResMLPLayer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor):
        super().__init__()
        self.cl = CommunicationLayer(num_features, num_patches)
        self.ff = FeedForward(num_features, expansion_factor)

    def forward(self, x):
        x = self.cl(x)
        out = self.ff(x)
        return out


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class ResMLP(nn.Module):
    def __init__(
            self,
            image_size=256,
            patch_size=16,
            in_channels=3,
            num_features=128,
            expansion_factor=2,
            num_layers=8,
            num_classes=10,
    ):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__()
        self.patcher = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, stride=patch_size
        )
        self.mlps = nn.Sequential(
            *[
                ResMLPLayer(num_features, num_patches, expansion_factor)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_features, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_features)
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = self.mlps(patches)
        embedding = torch.mean(embedding, dim=1)
        logits = self.classifier(embedding)
        return logits


# In[ ]:


batch_size = 64  
num_classes = 10  
num_epochs = 10  
learning_rate = 0.01  

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])


trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

model = ResMLP(
    image_size=28,  # 图像大小
    patch_size=4,  # 块大小
    in_channels=1,
    num_features=128,  # 嵌入维度
    num_layers=6,  # 层数
    num_classes=num_classes  # 类别数
)



criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model)  


def train():
    model.train() 
    running_loss = 0.0  
    correct = 0  
    total = 
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data 
        optimizer.zero_grad()  

        outputs = model(inputs)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  

        running_loss += loss.item()  

        _, predicted = torch.max(outputs.data, 1) 
        total += labels.size(0) 
        correct += (predicted == labels).sum().item()  

        if i % 200 == 199:  
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    epoch_loss = running_loss / len(trainloader)  
    epoch_acc = 100 * correct / total 

    return epoch_loss, epoch_acc  


# 定义测试函数
def test():
    model.eval()  
    running_loss = 0.0  
    correct = 0  
    total = 0 
    with torch.no_grad():  
        for data in testloader:
            images, labels = data  
            outputs = model(images) 
            loss = criterion(outputs, labels)  

            running_loss += loss.item() 

            _, predicted = torch.max(outputs.data, 1)  

            total += labels.size(0) 
            correct += (predicted == labels).sum().item()  

    epoch_loss = running_loss / len(testloader)  
    epoch_acc = 100 * correct / total  

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        epoch_acc))

    return epoch_loss, epoch_acc  



test_losses = []  
test_accs = []  
for epoch in range(num_epochs):
    train() 
    test_loss, test_acc = test() 
    test_losses.append(test_loss)  
    test_accs.append(test_acc)  



def plot(losses, accs):
    plt.figure(figsize=(5, 5)) 
    plt.plot(losses) 
    plt.xlabel('Epoch')  
    plt.ylabel('Loss')  
    plt.title('Loss Curve')  
    plt.savefig('./picture/resmlp_loss.png')  
    plt.show()  

    plt.figure(figsize=(5, 5))  
    # plt.subplot(1, 2, 2) 
    plt.plot(accs)  
    plt.xlabel('Epoch')  
    plt.ylabel('Accuracy (%)')  
    plt.title('Accuracy Curve') 
    plt.savefig('./picture/resmlp_accuracy.png')  
    plt.show()  

plot(test_losses, test_accs)  

