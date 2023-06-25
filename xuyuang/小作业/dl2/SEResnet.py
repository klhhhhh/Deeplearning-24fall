#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) 
        y = self.fc(y).view(b, c, 1, 1)  
        return x * y.expand_as(x)  


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes=10):
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


# In[2]:


transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)  
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)  

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)  
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)  

model = se_resnet18()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 随机梯度下降


def train(epoch):
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device) 
        optimizer.zero_grad() 
        outputs = model(inputs) 
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item() 
        if (i+1) % 200 == 0: 
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

def test():
    model.eval() 
    correct = 0 
    total = 0 
    running_loss = 0.0  
    with torch.no_grad(): # 不计算梯度
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device) 
            outputs = model(inputs) 
            loss = criterion(outputs, labels)  
            running_loss += loss.item()  
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别

            total += labels.size(0)  # 累加样本数
            correct += (predicted == labels).sum().item()  # 正确预测数

    epoch_loss = running_loss / len(test_loader)  
    epoch_acc = 100 * correct / total  
    return epoch_loss, epoch_acc  


test_losses = []  
test_accs = []  
epochs = 20
for epoch in range(epochs):
    train(epoch)  
    test_loss, test_acc = test()  
    print(f"{test_loss}  {test_acc}")
    test_losses.append(test_loss)  
    test_accs.append(test_acc)  

plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, epochs + 1), test_losses)
plt.title('loss')

plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, epochs + 1), test_accs)
plt.title('accuracy')

