from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EVFlowNetLike(nn.Module):
    """Small U-Net-like flow decoder inspired by EV-FlowNet.

    This is intentionally lightweight so it can train on CPU for local fixture
    tests while keeping the shape of a shared learned decoder for the real
    benchmark.
    """

    def __init__(self, in_channels: int, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)

        self.head = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        b = self.bottleneck(self.pool2(s2))

        d2 = F.interpolate(b, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)
        return self.head(d1)
