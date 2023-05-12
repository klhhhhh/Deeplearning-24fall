import torch
import torch.nn as nn
import torch.nn.functional as F

class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv_list = nn.ModuleList([])
        for i in range(M):
            self.conv_list.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, out_channels * M)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        output_list = []
        for i, conv in enumerate(self.conv_list):
            output_list.append(conv(x))
        U = torch.stack(output_list, dim=1)  # (batch_size, 1, out_channels, H, W)
        U1 = torch.sum(U, dim=1)
        S = self.global_pool(U1).view(batch_size, 1, 1, -1)  # (batch_size, 1, 1, out_channels)
        A = self.softmax(self.fc(S)).view(batch_size, self.M, self.out_channels, 1,
                                          1)  # (batch_size, M, out_channels, 1, 1)
        attentioned_feature_map = torch.sum(U * A, dim=1)  # (batch_size, out_channels, H, W)
        return attentioned_feature_map

class SKNet(nn.Module):
    def __init__(self, num_classes=100):
        super(SKNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.skconv1 = SKConv(64, 64)
        self.skconv2 = SKConv(64, 128, stride=2)
        self.skconv3 = SKConv(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.skconv1(out)
        out = self.skconv2(out)
        out = self.skconv3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

