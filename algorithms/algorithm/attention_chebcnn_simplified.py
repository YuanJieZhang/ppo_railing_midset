import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Simplified_DSTAGNN(nn.Module):
    def __init__(self, DEVICE, in_channels, out_channels, heads, time_steps=10):
        super(Simplified_DSTAGNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.attention = MultiHeadAttention(DEVICE, out_channels, heads)
        self.gtu = GTU(out_channels, time_strides=1, kernel_size=3)
        self.DEVICE = DEVICE

    def forward(self, x):
        x = x.to(self.DEVICE)

        # 确保输入是4维
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)  # (batch_size, in_channels, 1, 1)

        x = self.conv(x)  # 卷积期望输入形状 (batch_size, channels, height, width)

        # 调整输入形状以适配注意力机制
        if len(x.shape) == 4:
            x = x.squeeze(3).squeeze(2)  # (batch_size, out_channels)

        x = self.attention(x)  # 通过注意力机制

        # 确保 GTU 的输入是4D
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)  # (batch_size, out_channels, 1, 1)

        x = self.gtu(x)  # GTU 期望 4D 输入 (batch_size, out_channels, height, width)

        # 返回 2D 形状
        if len(x.shape) == 4:
            x = x.squeeze(3).squeeze(2)  # (batch_size, out_channels)

        return x

class GTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))
        self.fc = nn.Linear(in_channels, in_channels)  # 全连接层用于特征映射

    def forward(self, x):
        # 确保输入是4维
        if len(x.shape) == 3:
            x = x.unsqueeze(2)  # (batch_size, channels, 1, width)
        elif len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)  # (batch_size, channels, 1, 1)

        # 如果输入的通道数不等于 in_channels，使用全连接层进行特征映射
        if x.shape[1] != self.in_channels:
            x = x.view(x.size(0), -1)  # 展平除了batch_size的其他维度
            x = self.fc(x)
            x = x.view(x.size(0), self.in_channels, 1, 1)  # 重塑为原始形状
        if x.shape[3]<self.con2out.kernel_size[1]:
            padding = self.con2out.kernel_size[1]-x.shape[3]
            x = F.pad(x, (0, padding))

        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, :self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu

class MultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, features, heads):
        super(MultiHeadAttention, self).__init__()
        self.query = nn.Linear(features, features)
        self.key = nn.Linear(features, features)
        self.value = nn.Linear(features, features)
        self.fc_out = nn.Linear(features, features)
        self.d_k = features // heads
        self.heads = heads

    def forward(self, x):
        batch_size = x.size(0)

        # 线性 -> 视图重建/reshape -> 转置
        query = self.query(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)

        out = self.fc_out(out)

        return out