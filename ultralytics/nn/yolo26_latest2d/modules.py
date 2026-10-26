"""Lightweight YOLO26 plug-ins inspired by recent 2D object detectors.

These modules are intentionally small adapters rather than full detector
re-implementations. They keep the normal YOLO26 YAML contract:
    module(c1, c2, *args) -> feature map with c2 channels.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv


class _ChannelSpatialGate(nn.Module):
    """Shared channel and spatial reweighting helper."""

    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        hidden = max(c // reduction, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        avg = x.mean(1, keepdim=True)
        maxv = x.amax(1, keepdim=True)
        return x * self.spatial(torch.cat((avg, maxv), 1))


class FTFSODReweight(nn.Module):
    """Feature reweighting block inspired by FT-FSOD hybrid decoder diversity."""

    def __init__(self, c1: int, c2: int, branches: int = 3, reduction: int = 4):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.shared = Conv(c2, c2, 1, 1)
        self.branches = nn.ModuleList(DWConv(c2, c2, 3 + 2 * i, 1) for i in range(max(branches, 1)))
        self.gate = _ChannelSpatialGate(c2, reduction)
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        shared = self.shared(x)
        ensemble = torch.stack([branch(shared) for branch in self.branches], 0).mean(0)
        return x + self.out(self.gate(ensemble))


class YOLOv10CIBLite(nn.Module):
    """Compact inverted block inspired by YOLOv10 efficiency-driven CIB design."""

    def __init__(self, c1: int, c2: int, expansion: float = 1.5, shortcut: bool = True):
        super().__init__()
        hidden = max(int(c2 * expansion), 8)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.dw = DWConv(hidden, hidden, 3, 1)
        self.pw = Conv(hidden, c2, 1, 1, act=False)
        self.attn = _ChannelSpatialGate(c2, 8)
        self.add = shortcut and c1 == c2
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attn(self.pw(self.dw(self.cv1(x))))
        return self.act(x + y) if self.add else self.act(y)


class RTDETRv2HybridEncoder(nn.Module):
    """Selective multi-scale encoder inspired by RT-DETRv2 decoder sampling."""

    def __init__(self, c1: int, c2: int, reduction: int = 4):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1)
        self.local = DWConv(c2, c2, 3, 1)
        self.mid = DWConv(c2, c2, 5, 1)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, max(c2 // reduction, 8), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c2 // reduction, 8), c2 * 3, 1, bias=True),
        )
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        feats = torch.stack((x, self.local(x), self.mid(x)), 1)
        weights = self.context(x).view(x.shape[0], 3, x.shape[1], 1, 1).softmax(1)
        return self.out((feats * weights).sum(1))


class DFINEDistributionRefine(nn.Module):
    """Feature-level distribution refinement inspired by D-FINE FDR."""

    def __init__(self, c1: int, c2: int, bins: int = 8):
        super().__init__()
        self.bins = max(int(bins), 2)
        self.proj = Conv(c1, c2, 1, 1)
        self.logits = nn.Conv2d(c2, c2 * self.bins, 1, bias=True)
        self.refine = Conv(c2, c2, 3, 1)
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, c, h, w = x.shape
        prob = self.logits(x).view(b, c, self.bins, h, w).softmax(2)
        grid = torch.linspace(0, 1, self.bins, device=x.device, dtype=x.dtype).view(1, 1, self.bins, 1, 1)
        expectation = (prob * grid).sum(2)
        return self.out(x + self.refine(x * (1.0 + expectation)))


class LWDETRTokenMixer(nn.Module):
    """Light token mixer inspired by LW-DETR interleaved local/global attention."""

    def __init__(self, c1: int, c2: int, kernel: int = 7, reduction: int = 4):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1)
        self.local = DWConv(c2, c2, kernel, 1)
        hidden = max(c2 // reduction, 8)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.out(self.local(x) + x * self.global_gate(x))


class YOLOv12AreaAttention(nn.Module):
    """Area-token attention inspired by YOLOv12 attention-centric detector blocks."""

    def __init__(self, c1: int, c2: int, areas: int = 4, heads: int = 4):
        super().__init__()
        self.areas = max(int(areas), 2)
        self.heads = max(int(heads), 1)
        self.proj = Conv(c1, c2, 1, 1)
        self.q = nn.Conv2d(c2, c2, 1, bias=False)
        self.k = nn.Conv2d(c2, c2, 1, bias=False)
        self.v = nn.Conv2d(c2, c2, 1, bias=False)
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, c, h, w = x.shape
        heads = math.gcd(self.heads, c)
        d = c // heads
        q = self.q(x).view(b, heads, d, h * w).transpose(-1, -2)
        pooled = F.adaptive_avg_pool2d(x, (self.areas, self.areas))
        k = self.k(pooled).view(b, heads, d, -1)
        v = self.v(pooled).view(b, heads, d, -1).transpose(-1, -2)
        attn = (q @ k) * (d**-0.5)
        y = (attn.softmax(-1) @ v).transpose(-1, -2).reshape(b, c, h, w)
        return self.out(x + y)


class YOLOERepRTA(nn.Module):
    """Prompt-prototype adapter inspired by YOLOE RepRTA/LRPC alignment."""

    def __init__(self, c1: int, c2: int, prompts: int = 16):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1)
        self.prompts = nn.Parameter(torch.randn(max(int(prompts), 1), c2) * 0.02)
        self.region = Conv(c2, c2, 3, 1)
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, c, h, w = x.shape
        q = F.normalize(self.region(x).flatten(2).transpose(1, 2), dim=-1)
        p = F.normalize(self.prompts, dim=-1)
        gate = (q @ p.t()).amax(-1).view(b, 1, h, w).sigmoid()
        return self.out(x * (1.0 + gate))


class PKIBlock(nn.Module):
    """Poly-kernel inception block inspired by PKINet and CAA context anchoring."""

    def __init__(self, c1: int, c2: int, kernels: tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1)
        self.branches = nn.ModuleList(DWConv(c2, c2, k, 1) for k in kernels)
        self.anchor_h = nn.Conv2d(c2, c2, (1, 7), padding=(0, 3), groups=c2, bias=False)
        self.anchor_v = nn.Conv2d(c2, c2, (7, 1), padding=(3, 0), groups=c2, bias=False)
        self.mix = Conv(c2 * (len(kernels) + 2), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        context = self.anchor_v(self.anchor_h(x))
        return self.mix(torch.cat([x, context, *[branch(x) for branch in self.branches]], 1))


class MambaYOLORGBlock(nn.Module):
    """Linear-complexity gated spatial mixer inspired by Mamba-YOLO RG blocks."""

    def __init__(self, c1: int, c2: int, kernel: int = 7):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1)
        self.in_proj = nn.Conv2d(c2, c2 * 2, 1, bias=True)
        self.scan_h = nn.Conv2d(c2, c2, (1, kernel), padding=(0, kernel // 2), groups=c2, bias=False)
        self.scan_v = nn.Conv2d(c2, c2, (kernel, 1), padding=(kernel // 2, 0), groups=c2, bias=False)
        self.channel = Conv(c2, c2, 1, 1)
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        u, gate = self.in_proj(x).chunk(2, 1)
        y = self.scan_h(u) + self.scan_v(u)
        y = self.channel(y) * gate.sigmoid()
        return self.out(x + y)


class RFDETRNASBlock(nn.Module):
    """Small mixed-operation block inspired by RF-DETR architecture search."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1)
        self.ops = nn.ModuleList(
            (
                DWConv(c2, c2, 3, 1),
                DWConv(c2, c2, 5, 1),
                LWDETRTokenMixer(c2, c2, 5),
                MambaYOLORGBlock(c2, c2, 5),
            )
        )
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        weights = self.alpha.softmax(0)
        y = sum(w * op(x) for w, op in zip(weights, self.ops))
        return self.out(x + y)
