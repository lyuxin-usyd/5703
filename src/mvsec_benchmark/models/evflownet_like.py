from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _init_paper_conv(conv: nn.Conv2d, *, gain: float = 0.1) -> None:
    nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
    conv.weight.data.mul_(gain)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def _paper_batch_norm(channels: int) -> nn.BatchNorm2d:
    bn = nn.BatchNorm2d(channels, eps=1e-5)
    nn.init.constant_(bn.weight, 0.01)
    nn.init.zeros_(bn.bias)
    return bn


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


class StridedConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 2, batch_norm: bool = False) -> None:
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        _init_paper_conv(conv)
        layers: list[nn.Module] = [conv]
        if batch_norm:
            layers.append(_paper_batch_norm(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, batch_norm: bool = False) -> None:
        super().__init__()
        self.conv1 = StridedConv(channels, channels, stride=1, batch_norm=batch_norm)
        conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        _init_paper_conv(conv)
        layers: list[nn.Module] = [conv]
        if batch_norm:
            layers.append(_paper_batch_norm(channels))
        layers.append(nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + x


class UpsampleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, batch_norm: bool = False) -> None:
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        _init_paper_conv(conv)
        layers: list[nn.Module] = [nn.ReflectionPad2d(1), conv]
        if batch_norm:
            layers.append(_paper_batch_norm(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_size, mode="nearest")
        return self.block(x)


class FlowHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, 2, kernel_size=1)
        _init_paper_conv(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(x)) * 256.0


class EVFlowNetMultiScale(nn.Module):
    """PyTorch EV-FlowNet-style decoder with four flow scales.

    This follows the public EV-FlowNet model structure at a practical level:
    four stride-2 encoder convolutions, two residual transition blocks, decoder
    skip connections, one predicted flow per decoder scale, and the final
    full-resolution flow as the last output.
    """

    def __init__(self, in_channels: int, base_channels: int = 64, *, batch_norm: bool = False) -> None:
        super().__init__()
        b = int(base_channels)
        if b < 8:
            raise ValueError("base_channels must be >= 8 for EVFlowNetMultiScale.")

        self.enc0 = StridedConv(in_channels, b, stride=2, batch_norm=batch_norm)
        self.enc1 = StridedConv(b, b * 2, stride=2, batch_norm=batch_norm)
        self.enc2 = StridedConv(b * 2, b * 4, stride=2, batch_norm=batch_norm)
        self.enc3 = StridedConv(b * 4, b * 8, stride=2, batch_norm=batch_norm)

        self.res0 = ResidualBlock(b * 8, batch_norm=batch_norm)
        self.res1 = ResidualBlock(b * 8, batch_norm=batch_norm)

        self.dec0 = UpsampleConv(b * 16, b * 4, batch_norm=batch_norm)
        self.flow0 = FlowHead(b * 4)
        self.dec1 = UpsampleConv(b * 8 + 2, b * 2, batch_norm=batch_norm)
        self.flow1 = FlowHead(b * 2)
        self.dec2 = UpsampleConv(b * 4 + 2, b, batch_norm=batch_norm)
        self.flow2 = FlowHead(b)
        self.dec3 = UpsampleConv(b * 2 + 2, max(b // 2, 8), batch_norm=batch_norm)
        self.flow3 = FlowHead(max(b // 2, 8))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        input_size = x.shape[-2:]
        skip0 = self.enc0(x)
        skip1 = self.enc1(skip0)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)

        y = self.res1(self.res0(skip3))

        y = torch.cat([y, skip3], dim=1)
        y = self.dec0(y, target_size=skip2.shape[-2:])
        flow0 = self.flow0(y)

        y = torch.cat([y, flow0, skip2], dim=1)
        y = self.dec1(y, target_size=skip1.shape[-2:])
        flow1 = self.flow1(y)

        y = torch.cat([y, flow1, skip1], dim=1)
        y = self.dec2(y, target_size=skip0.shape[-2:])
        flow2 = self.flow2(y)

        y = torch.cat([y, flow2, skip0], dim=1)
        y = self.dec3(y, target_size=input_size)
        flow3 = self.flow3(y)
        return [flow0, flow1, flow2, flow3]
