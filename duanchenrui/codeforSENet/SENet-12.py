import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR100(root='./data100', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data100', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat',
 #          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone',
           90: 'train', 28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 
           82: 'sunflower', 17: 'castle', 71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 
           74: 'shrew', 59: 'pine_tree', 70: 'rose', 87: 'television', 84: 'table', 64: 'possum', 
           52: 'oak_tree', 42: 'leopard', 47: 'maple_tree', 65: 'rabbit', 21: 'chimpanzee', 
           22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster', 49: 'mountain', 
           56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 
           6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 
           35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange', 92: 'tulip', 50: 'mouse', 
           15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail', 
           69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 
           34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 
           26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 
           2: 'baby', 41: 'lawn_mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 
           60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}

import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation block
class SE_Block(nn.Module):                         
    def __init__(self, in_channel):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channel, in_channel // 8, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel // 8, in_channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out


# SE残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.channel = in_channel
        self.res1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),  # 保持尺寸不变
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.SE = SE_Block(out_channel)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:      # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )


    def forward(self, x):
        out = self.res1(x)
        SE_out = self.SE(out)
        out = out * SE_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 残差网络
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3,stride=1, padding=1, bias=False)  #[3,32,32] -> [8,32,32]                
        self.bn1 = nn.BatchNorm2d(8)
        self.reslayer1 = ResidualBlock(8,8) # [8,32,32]
        self.reslayer2 = ResidualBlock(8,16,2) # [8,32,32] -> [16,16,16]
        self.reslayer3 = ResidualBlock(16,16)  # [16,16,16]
        self.reslayer4 = ResidualBlock(16,32,2) # [16,16,16] -> [32,8,8]
        self.reslayer5 = ResidualBlock(32,32) # [32,8,8]
        self.pool = nn.AvgPool2d(2) # [32,8,8]->[32,4,4]
        self.fc = nn.Linear(32 * 4 * 4, 100)
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.reslayer1(out)
        out = self.reslayer2(out)
        out = self.reslayer3(out)
        out = self.reslayer4(out)
        out = self.reslayer5(out)
        out = self.pool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out


net = ResNet()
print(net)
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



def train(epoch):
    print("yes")
#for epoch in range(2):  # loop over the dataset multiple times
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    print('Finished Training epoch %d'%(epoch+1))

def validate(loss_vector, accuracy_vector):
    net.eval()
    total = 0
    val_loss, correct = 0, 0
    for data, target in testloader:
        output = net(data)
        val_loss += criterion(output, target).data.item()
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    val_loss /= total
    loss_vector.append(val_loss)

    accuracy = 100* correct/ total
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, total, accuracy))

epochs = 10

lossv, accv = [], []

for epoch in range(epochs):  # loop over the dataset multiple times

    train(epoch)
    validate(lossv, accv)

print('Finished Training')