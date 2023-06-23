import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from data.dataset import MyData


from models.SKNet_van1 import SKAttentionNet
from models.SKNet_van2 import AttentionSKNet


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


# 模型训练的函数
def model_training(model, train_loader, test_loader, optimizer, criterion, num_epochs=100, model_name='model.pth'):
    # 记录每个epoch的训练和测试损失以及准确率
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

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
            torch.save(model.state_dict(), f'best_{model_name}')
            best_acc = test_acc
    return train_losses, train_accs, test_losses, test_accs


if __name__ == "__main__":
    # 检查是否有可用的GPU，如果有，使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MyData(train=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataset = MyData(train=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss().to(device)

    num_epochs = 100

    # 第二个模型
    model2 = AttentionSKNet().to(device)  # 请确保SKNet模型已经被定义
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-5)
    train_losses2, train_accs2, test_losses2, test_accs2 = model_training(model2, train_loader, test_loader, optimizer2,
                                                                          criterion, model_name='CombinedNet.pth')

    # 绘制损失图像
    plt.figure()
    plt.plot(range(num_epochs), train_losses2, label='Train Loss')
    plt.plot(range(num_epochs), test_losses2, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('CombinedNet_loss.png')  # 保存图像

    # 绘制准确率图像
    plt.figure()
    plt.plot(range(num_epochs), train_accs2, label='Train Accuracy')
    plt.plot(range(num_epochs), test_accs2, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('CombinedNet_accuracy.png')  # 保存图像


    # 第一个模型
    model1 = SKAttentionNet().to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001, weight_decay=1e-5)
    train_losses1, train_accs1, test_losses1, test_accs1 = model_training(model1, train_loader, test_loader, optimizer1, criterion, model_name='SKAttentionNet.pth')

    # 绘制损失图像
    plt.figure()
    plt.plot(range(num_epochs), train_losses1, label='Train Loss')
    plt.plot(range(num_epochs), test_losses1, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('SKAttentionNet_loss.png')  # 保存图像

    # 绘制准确率图像
    plt.figure()
    plt.plot(range(num_epochs), train_accs1, label='Train Accuracy')
    plt.plot(range(num_epochs), test_accs1, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('SKAttentionNet_accuracy.png')  # 保存图像

