#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 attention modules and CNN extractor."""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ChannelAttention(nn.Module):
    """CBAM Channel Attention: 学习通道间的重要性权重。"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        # Global Average Pool + Global Max Pool
        avg_out = x.view(b, c, -1).mean(dim=2)  # (B, C)
        max_out = x.view(b, c, -1).max(dim=2)[0]  # (B, C)
        # 共享MLP
        att = torch.sigmoid(self.mlp(avg_out) + self.mlp(max_out))  # (B, C)
        return x * att.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention: 学习空间位置的重要性权重。"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        max_out = x.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        att = torch.sigmoid(self.conv(cat))  # (B, 1, H, W)
        return x * att


class CBAMBlock(nn.Module):
    """CBAM: Channel Attention → Spatial Attention (串联)。"""

    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class AttentionCNN(BaseFeaturesExtractor):
    """
    CNN + CBAM 注意力特征提取器。

    架构: 3层Conv2d(对齐v11 LightweightCNN尺寸) + 每层后CBAM + 128隐藏层 + 64输出
    参数增量: ~2000 (CBAM模块)，Jetson Nano友好
    """

    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        print(f"\n🧠 V14 AttentionCNN (CNN + CBAM):")
        print(f"   输入: {n_input_channels}通道 x {observation_space.shape[1]}x{observation_space.shape[2]}")

        # Conv backbone (与LightweightCNN相同卷积参数)
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.cbam1 = CBAMBlock(32, reduction=4, spatial_kernel=7)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.cbam2 = CBAMBlock(64, reduction=8, spatial_kernel=7)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.cbam3 = CBAMBlock(64, reduction=8, spatial_kernel=5)

        self.flatten = nn.Flatten()

        # 计算flatten维度
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            x = torch.relu(self.conv1(sample))
            x = self.cbam1(x)
            x = torch.relu(self.conv2(x))
            x = self.cbam2(x)
            x = torch.relu(self.conv3(x))
            x = self.cbam3(x)
            n_flatten = self.flatten(x).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

        total_params = sum(p.numel() for p in self.parameters())
        cbam_params = sum(
            p.numel() for m in [self.cbam1, self.cbam2, self.cbam3] for p in m.parameters()
        )
        print(f"   展平维度: {n_flatten}")
        print(f"   特征维度: {features_dim}")
        print(f"   CBAM参数: {cbam_params:,}")
        print(f"   总参数: {total_params:,}")

    def forward(self, observations):
        x = torch.relu(self.conv1(observations))
        x = self.cbam1(x)
        x = torch.relu(self.conv2(x))
        x = self.cbam2(x)
        x = torch.relu(self.conv3(x))
        x = self.cbam3(x)
        return self.linear(self.flatten(x))


__all__ = [
    "ChannelAttention",
    "SpatialAttention",
    "CBAMBlock",
    "AttentionCNN",
]
