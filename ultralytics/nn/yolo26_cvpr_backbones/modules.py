"""Detector-friendly CVPR backbone adapters for YOLO26.

The modules in this file are compact PyTorch adaptations of ideas from:
1. LSNet: See Large, Focus Small, CVPR 2025.
2. AKCMamba-YOLO: Selective State Space Models For Real-Time Object Detection, CVPR 2026.
3. Convolutional Neural Networks Driven by Content Similarity, CVPR 2026.

They are not full paper re-implementations.  They keep the Ultralytics parser
contract: module(c1, c2, n, *args) -> tensor with c2 channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv


class LayerScale2d(nn.Module):
    """Small learnable residual scale used to avoid early training shocks."""

    def __init__(self, channels: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class LSConvBlock(nn.Module):
    """Large-small receptive-field convolution block inspired by LSNet."""

    def __init__(self, channels: int, large_kernel: int = 7, small_kernel: int = 3, expansion: float = 2.0):
        super().__init__()
        hidden = max(int(channels * float(expansion)), 16)
        large_kernel = max(int(large_kernel), 3)
        small_kernel = max(int(small_kernel), 3)
        if large_kernel % 2 == 0:
            large_kernel += 1
        if small_kernel % 2 == 0:
            small_kernel += 1

        self.large = nn.Conv2d(
            channels,
            channels,
            large_kernel,
            padding=large_kernel // 2,
            groups=channels,
            bias=False,
        )
        self.small = nn.Conv2d(
            channels,
            channels,
            small_kernel,
            padding=small_kernel // 2,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.channel = nn.Sequential(
            Conv(channels, hidden, 1, 1),
            Conv(hidden, channels, 1, 1, act=False),
        )
        self.scale = LayerScale2d(channels, 1e-2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn(self.large(x) + self.small(x))
        y = self.channel(self.act(y))
        return x + self.scale(y)


class LSNetStage(nn.Module):
    """CSP-wrapped LSNet adapter for replacing YOLO26 C3k2 stages."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        expansion: float = 0.5,
        large_kernel: int = 7,
        small_kernel: int = 3,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        hidden = max(int(c2 * float(expansion)), 16)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.blocks = nn.Sequential(
            *[
                LSConvBlock(hidden, large_kernel=large_kernel, small_kernel=small_kernel, expansion=mlp_ratio)
                for _ in range(max(int(n), 1))
            ]
        )
        self.cv3 = Conv(hidden * 2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), 1))


class AKCMambaBlock(nn.Module):
    """Adaptive-kernel cross-scan mixer inspired by AKCMamba-YOLO."""

    def __init__(self, channels: int, kernels: tuple[int, int] = (5, 9), expansion: float = 1.0):
        super().__init__()
        hidden = max(int(channels * float(expansion)), 16)
        k1, k2 = int(kernels[0]), int(kernels[1])
        if k1 % 2 == 0:
            k1 += 1
        if k2 % 2 == 0:
            k2 += 1

        self.in_proj = Conv(channels, hidden * 2, 1, 1)
        self.scan_h = nn.Conv2d(hidden, hidden, (1, k2), padding=(0, k2 // 2), groups=hidden, bias=False)
        self.scan_v = nn.Conv2d(hidden, hidden, (k2, 1), padding=(k2 // 2, 0), groups=hidden, bias=False)
        self.local = nn.Conv2d(hidden, hidden, k1, padding=k1 // 2, groups=hidden, bias=False)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, max(hidden // 4, 8), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(hidden // 4, 8), hidden * 3, 1, bias=True),
        )
        self.out = Conv(hidden, channels, 1, 1, act=False)
        self.scale = LayerScale2d(channels, 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, gate = self.in_proj(x).chunk(2, 1)
        branches = torch.stack((self.local(u), self.scan_h(u), self.scan_v(u)), 1)
        weights = self.router(u).view(u.shape[0], 3, u.shape[1], 1, 1).softmax(1)
        y = (branches * weights).sum(1) * gate.sigmoid()
        return x + self.scale(self.out(y))


class AKCMambaStage(nn.Module):
    """CSP-wrapped adaptive cross-scan state-space-style YOLO26 stage."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        expansion: float = 0.5,
        small_kernel: int = 5,
        large_kernel: int = 9,
        scan_ratio: float = 1.0,
    ):
        super().__init__()
        hidden = max(int(c2 * float(expansion)), 16)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.blocks = nn.Sequential(
            *[
                AKCMambaBlock(hidden, kernels=(small_kernel, large_kernel), expansion=scan_ratio)
                for _ in range(max(int(n), 1))
            ]
        )
        self.cv3 = Conv(hidden * 2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), 1))


class EgoCSBlock(nn.Module):
    """Content-similarity convolution block inspired by Ego/CNN driven by similarity."""

    def __init__(self, channels: int, pool_size: int = 7, reduction: int = 4):
        super().__init__()
        kernel = max(int(pool_size), 3)
        if kernel % 2 == 0:
            kernel += 1
        self.norm = nn.BatchNorm2d(channels)
        self.local = DWConv(channels, channels, 3, 1)
        self.content = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // max(int(reduction), 1), 8), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(channels // max(int(reduction), 1), 8), channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out = Conv(channels, channels, 1, 1, act=False)
        self.scale = LayerScale2d(channels, 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        key = x_norm.mean(1).flatten(1)
        order = key.argsort(dim=1)
        inverse = order.argsort(dim=1)
        flat = x_norm.flatten(2)
        sorted_flat = flat.gather(2, order.unsqueeze(1).expand(-1, c, -1))
        content = self.content(sorted_flat)
        content = content.gather(2, inverse.unsqueeze(1).expand(-1, c, -1)).reshape(b, c, h, w)
        y = self.out(self.local(x_norm) + content * self.gate(x_norm))
        return x + self.scale(y)


class EgoCSStage(nn.Module):
    """CSP-wrapped content-similarity stage for YOLO26 backbones."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        expansion: float = 0.5,
        pool_size: int = 7,
        reduction: int = 4,
    ):
        super().__init__()
        hidden = max(int(c2 * float(expansion)), 16)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.blocks = nn.Sequential(
            *[EgoCSBlock(hidden, pool_size=pool_size, reduction=reduction) for _ in range(max(int(n), 1))]
        )
        self.cv3 = Conv(hidden * 2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), 1))
