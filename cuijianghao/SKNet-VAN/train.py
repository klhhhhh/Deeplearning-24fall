import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from data.dataset import MyData


#from models.SKNet import SKNet
#from models.SKNet_pro import SKNet

from models.SKNet_van import SKNetVAN


# 训练函数
def train(model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels).item()
    epoch_loss = train_loss / len(train_loader.dataset)
    epoch_acc = train_acc / len(train_loader.dataset)
    return epoch_loss, epoch_acc


# 测试函数
def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == labels).item()
            predicted_labels.extend(preds.tolist())
            true_labels.extend(labels.tolist())
    report = classification_report(true_labels, predicted_labels, zero_division=1)
    print(report)
    epoch_loss = test_loss / len(test_loader.dataset)
    epoch_acc = test_acc / len(test_loader.dataset)
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    # 检查是否有可用的GPU，如果有，使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MyData(train=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataset = MyData(train=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = SKNetVAN().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 记录每个epoch的训练和测试损失以及准确率
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    num_epochs = 100
    best_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, optimizer, criterion, train_loader)
        test_loss, test_acc = test(model, criterion, test_loader)

        # 记录结果
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
              .format(epoch + 1, num_epochs, train_loss, train_acc, test_loss, test_acc))

        # 每个epoch保存一次模型
        if test_acc > best_acc:
            torch.save(model.state_dict(), f'best_model.pth')
            best_acc = test_acc

    # 绘制损失图像
    plt.figure()
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')  # 保存图像

    # 绘制准确率图像
    plt.figure()
    plt.plot(range(num_epochs), train_accs, label='Train Accuracy')
    plt.plot(range(num_epochs), test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')  # 保存图像