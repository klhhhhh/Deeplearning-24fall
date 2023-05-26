import torch
import torch.nn as nn


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


class VisualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(VisualAttentionBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        N, C, H, W = x.size()
        x_pool = torch.mean(x, dim=[2, 3])
        x_pool = x_pool.view(N, C)
        attention_weights = self.fc(x_pool)
        attention_weights = torch.softmax(attention_weights, dim=0)
        x_att = x * attention_weights.view(N, C, 1, 1)
        return x_att


class SKConvAttention(nn.Module):
    def __init__(self, in_channels, out_channels, M=2, r=16, L=32, dropout_rate=0.5):
        super(SKConvAttention, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv_list = nn.ModuleList([])
        self.dropout = nn.Dropout(dropout_rate)

        for i in range(M):
            self.conv_list.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1 + i, dilation=1 + i,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        self.fc1 = nn.Linear(out_channels, d)
        self.fc2 = nn.Linear(d, out_channels * M)

        self.softmax = nn.Softmax(dim=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Attention
        self.fc_attention = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        batch_size = x.size(0)
        output_list = []
        for i, conv in enumerate(self.conv_list):
            output_list.append(conv(x))
        U = torch.stack(output_list, dim=1)
        U1 = torch.sum(U, dim=1)
        S = self.global_pool(U1).view(batch_size, 1, 1, -1)
        A = self.softmax(self.fc2(self.fc1(S))).view(batch_size, self.M, self.out_channels, 1, 1)

        # Attention
        pooled_features = torch.mean(U1, dim=[2, 3]).view(batch_size, self.out_channels)
        attention_weights = torch.softmax(self.fc_attention(pooled_features), dim=0)

        attentioned_feature_map = torch.sum(U * A, dim=1) * attention_weights.view(batch_size, self.out_channels, 1, 1)
        return self.dropout(attentioned_feature_map)


class SKAttentionNet(nn.Module):
    def __init__(self, num_classes=100):
        super(SKAttentionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.skconvatt1 = SKConvAttention(64, 128)
        self.skconvatt2 = SKConvAttention(128, 256)
        self.skconvatt3 = SKConvAttention(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.skconvatt1(out)
        out = self.skconvatt2(out)
        out = self.skconvatt3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out