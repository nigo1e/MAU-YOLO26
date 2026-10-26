"""Focused YOLO26 modules inspired by HVI, MSHC, StarNet and sMLP.

The classes keep Ultralytics YAML compatibility:
    module(c1, c2, *args) -> feature map with c2 channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv


class HVIEnhanceStem(nn.Module):
    """Low-light enhancement stem inspired by HVI-CIDNet's HVI color decomposition."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, expand: int = 2):
        super().__init__()
        hidden = max(c2 * expand, 16)
        self.rgb_stem = Conv(c1, c2, k, s)
        self.hvi_stem = Conv(3, c2, k, s)
        self.fuse = nn.Sequential(
            Conv(c2 * 2, hidden, 1, 1),
            DWConv(hidden, hidden, 3, 1),
            Conv(hidden, c2, 1, 1),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, max(c2 // 4, 8), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c2 // 4, 8), c2, 1, bias=True),
            nn.Sigmoid(),
        )

    @staticmethod
    def _rgb_to_hvi(x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        maxc = x.max(1, keepdim=True).values
        minc = x.min(1, keepdim=True).values
        intensity = x.mean(1, keepdim=True)
        value = maxc
        saturation = (maxc - minc) / (maxc + 1e-6)
        warm_cool = (r - b) / (r + b + 1e-6)
        hue_vector = torch.cat((warm_cool, saturation, value), 1)
        return hue_vector * (0.5 + intensity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = self.rgb_stem(x)
        hvi = self.hvi_stem(self._rgb_to_hvi(x))
        y = self.fuse(torch.cat((rgb, hvi), 1))
        return y * (1.0 + self.gate(y))


class MSHCBlock(nn.Module):
    """Lightweight multi-scale spatial heterogeneous convolution block."""

    def __init__(self, c1: int, c2: int, kernels: tuple[int, ...] = (3, 5, 7), expansion: float = 0.5):
        super().__init__()
        hidden = max(int(c2 * expansion), 16)
        self.proj = Conv(c1, c2, 1, 1)
        self.reduce = Conv(c2, hidden, 1, 1)
        self.square = nn.ModuleList(DWConv(hidden, hidden, k, 1) for k in kernels)
        self.horizontal = nn.Conv2d(hidden, hidden, (1, 7), padding=(0, 3), groups=hidden, bias=False)
        self.vertical = nn.Conv2d(hidden, hidden, (7, 1), padding=(3, 0), groups=hidden, bias=False)
        self.fuse = Conv(hidden * (len(kernels) + 2), c2, 1, 1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, c2, 1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        h = self.reduce(x)
        feats = [branch(h) for branch in self.square]
        feats.extend((self.horizontal(h), self.vertical(h)))
        y = self.fuse(torch.cat(feats, 1))
        return x + y * self.gate(y)


class StarStem(nn.Module):
    """StarNet-style P1/2 stem.

    The first version downsampled twice inside the stem, which is convenient for
    classification but too aggressive for detection. Keeping a P1/2 feature
    gives the later P3/P4/P5 maps a better low-level signal.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class StarDown(nn.Module):
    """StarNet-style downsampling projection."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class StarBlock(nn.Module):
    """Detection-stable StarNet block.

    This follows the official StarNet block more closely: depthwise context,
    ReLU6-gated star multiplication, point projection, a second depthwise
    filter, then residual addition. A small residual scale keeps scratch
    detection training stable.
    """

    def __init__(self, c1: int, c2: int, mlp_ratio: float = 4.0, shortcut: bool = True, residual_scale: float = 0.1):
        super().__init__()
        hidden = max(int(c2 * mlp_ratio), 16)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.dwconv = nn.Sequential(
            nn.Conv2d(c2, c2, 7, padding=3, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.f1 = nn.Conv2d(c2, hidden, 1, bias=False)
        self.f2 = nn.Conv2d(c2, hidden, 1, bias=False)
        self.act = nn.ReLU6(inplace=True)
        self.g = nn.Sequential(
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.dwconv2 = nn.Conv2d(c2, c2, 7, padding=3, groups=c2, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        y = self.dwconv(x)
        y = self.act(self.f1(y)) * self.f2(y)
        y = self.dwconv2(self.g(y))
        return x + self.scale * y if self.add else y


class SMLPBlock(nn.Module):
    """Sparse/spatial MLP block using axial token mixing."""

    def __init__(self, c1: int, c2: int, expansion: float = 2.0):
        super().__init__()
        hidden = max(int(c2 * expansion), 16)
        self.proj = Conv(c1, c2, 1, 1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(c2, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, c2, 1, bias=False),
        )
        self.h_mix = nn.Conv1d(c2, c2, 7, padding=3, groups=c2, bias=False)
        self.w_mix = nn.Conv1d(c2, c2, 7, padding=3, groups=c2, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, c, h, w = x.shape
        h_token = x.mean(3)
        w_token = x.mean(2)
        h_gate = self.h_mix(h_token).sigmoid().view(b, c, h, 1)
        w_gate = self.w_mix(w_token).sigmoid().view(b, c, 1, w)
        y = x * (1.0 + h_gate + w_gate)
        y = self.channel_mlp(self.norm(y))
        return self.out(x + y)
