"""ECCV 2026 S2-FracMix and PriorEye adapters for YOLO26 main6.20.

These modules are designed for Ultralytics-style YAML parsing:
    module(c1, c2, *args) -> Tensor[B, c2, H, W]
or for multi-input fusion:
    module([c1, c2, ...], c_out, *args) -> Tensor[B, c_out, H, W].

They are detector-oriented adapters inspired by the papers, not drop-in copies
of the full original training pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv


class _LayerScale(nn.Module):
    """Tiny residual scale for stable insertion into pretrained-style YOLO blocks."""

    def __init__(self, c: int, init_value: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, c, 1, 1), float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _ChannelGate(nn.Module):
    """Lightweight channel gate shared by S2-FracMix and PriorEye adapters."""

    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        hidden = max(c // max(int(reduction), 1), 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class _VisualSpatialPrior(nn.Module):
    """Image-free visual prior built from coordinates, saliency, low-frequency context and edge energy."""

    def __init__(self, c: int, prior_channels: int = 16):
        super().__init__()
        pc = max(int(prior_channels), 8)
        self.prior = nn.Sequential(
            nn.Conv2d(6, pc, 3, padding=1, bias=False),
            nn.BatchNorm2d(pc),
            nn.SiLU(inplace=True),
            nn.Conv2d(pc, c, 1, bias=True),
            nn.Sigmoid(),
        )

    @staticmethod
    def _coords(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        rr = torch.sqrt(torch.clamp(xx.square() + yy.square(), min=1e-6))
        return torch.stack((xx, yy, rr), 0).unsqueeze(0).expand(b, -1, -1, -1)

    @staticmethod
    def _edge(x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(1, keepdim=True)
        dx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
        dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
        return dx.abs() + dy.abs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        saliency = x.mean(1, keepdim=True)
        low_freq = F.avg_pool2d(saliency, 7, stride=1, padding=3)
        prior = torch.cat((self._coords(x), saliency, low_freq, self._edge(x)), 1)
        return self.prior(prior)


class PriorEyeStem(nn.Module):
    """Backbone stem with PriorEye-style visual/spatial prior injection."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, prior_channels: int = 16):
        super().__init__()
        mid = max(c2 // 2, 16)
        self.rgb = Conv(c1, c2, k, s)
        self.prior_rgb = nn.Sequential(
            nn.Conv2d(c1 + 3, mid, k, s, k // 2, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.prior = _VisualSpatialPrior(c2, prior_channels)
        self.fuse = Conv(c2 * 2, c2, 1, 1)
        self.scale = _LayerScale(c2, 1e-2)

    @staticmethod
    def _rgb_hint(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        maxv = x.amax(1, keepdim=True)
        contrast = maxv - x.amin(1, keepdim=True)
        return torch.cat((mean, maxv, contrast), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.rgb(x)
        hint = self.prior_rgb(torch.cat((x, self._rgb_hint(x)), 1))
        gate = self.prior(base + hint)
        return base + self.scale(self.fuse(torch.cat((base * (1.0 + gate), hint * gate), 1)))


class PriorEyeBlock(nn.Module):
    """Prior-aware feature adapter for backbone, neck or detection pre-head."""

    def __init__(
        self,
        c1: int,
        c2: int,
        expansion: float = 0.5,
        prior_channels: int = 16,
        residual_scale: float = 1e-2,
    ):
        super().__init__()
        hidden = max(int(c2 * float(expansion)), 16)
        self.proj = Conv(c1, c2, 1, 1)
        self.local = nn.Sequential(DWConv(c2, hidden, 3, 1), Conv(hidden, c2, 1, 1))
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.prior = _VisualSpatialPrior(c2, prior_channels)
        self.out = Conv(c2, c2, 1, 1)
        self.scale = _LayerScale(c2, residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        y = self.local(x) * (1.0 + self.context(x)) * (1.0 + self.prior(x))
        return x + self.scale(self.out(y))


class PriorEyeC2f(nn.Module):
    """Complex backbone block: C2f split with repeated PriorEye memory-gated adapters."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        expansion: float = 0.5,
        prior_channels: int = 16,
        shortcut: bool = True,
    ):
        super().__init__()
        hidden = max(int(c2 * float(expansion)), 16)
        repeats = max(int(n), 1)
        self.cv1 = Conv(c1, hidden * 2, 1, 1)
        self.blocks = nn.ModuleList(PriorEyeBlock(hidden, hidden, 1.0, prior_channels, 1e-2) for _ in range(repeats))
        self.cv2 = Conv(hidden * (2 + repeats), c2, 1, 1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for block in self.blocks:
            y.append(block(y[-1]))
        out = self.cv2(torch.cat(y, 1))
        return x + out if self.shortcut else out


class S2FracMixBlock(nn.Module):
    """Feature-space S2-FracMix: self-saliency shape branches plus fractional scale bank."""

    def __init__(self, c1: int, c2: int, bins: int = 4, min_scale: float = 0.5, max_scale: float = 1.5):
        super().__init__()
        self.bins = max(int(bins), 2)
        self.scale_ratios = [
            float(min_scale) + (float(max_scale) - float(min_scale)) * i / max(self.bins - 1, 1)
            for i in range(self.bins)
        ]
        self.proj = Conv(c1, c2, 1, 1)
        self.shape_branches = nn.ModuleList(
            (
                DWConv(c2, c2, 3, 1),
                nn.Sequential(
                    nn.Conv2d(c2, c2, (1, 7), padding=(0, 3), groups=c2, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.SiLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(c2, c2, (7, 1), padding=(3, 0), groups=c2, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.SiLU(inplace=True),
                ),
            )
        )
        branch_count = len(self.shape_branches) + self.bins
        self.alpha = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, max(c2 // 4, 8), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c2 // 4, 8), branch_count, 1, bias=True),
        )
        self.channel = _ChannelGate(c2, 4)
        self.out = Conv(c2, c2, 1, 1)

    def _scale_bank(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        feats = []
        for ratio in self.scale_ratios:
            if abs(ratio - 1.0) < 1e-6:
                feats.append(x)
                continue
            resized = F.interpolate(
                x,
                size=(max(int(round(h * ratio)), 1), max(int(round(w * ratio)), 1)),
                mode="bilinear",
                align_corners=False,
            )
            feats.append(F.interpolate(resized, size=(h, w), mode="bilinear", align_corners=False))
        return torch.stack(feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b = x.shape[0]
        logits = self.alpha(x).view(b, -1, 1, 1, 1)
        shape_w = logits[:, : len(self.shape_branches)].softmax(1)
        scale_w = logits[:, len(self.shape_branches) :].softmax(1)
        shape_mix = (torch.stack([branch(x) for branch in self.shape_branches], 1) * shape_w).sum(1)
        scale_mix = (self._scale_bank(x) * scale_w).sum(1)
        return self.out(x + self.channel(shape_mix + scale_mix))


class S2FracMixFusion(nn.Module):
    """Multi-input S2-FracMix fusion for YOLO26 PAN/FPN neck nodes."""

    def __init__(
        self,
        channels: list[int] | tuple[int, ...],
        c2: int,
        bins: int = 4,
        min_scale: float = 0.5,
        max_scale: float = 1.5,
    ):
        super().__init__()
        self.c2 = int(c2)
        self.proj = nn.ModuleList(Conv(int(c), self.c2, 1, 1) for c in channels)
        self.level_logits = nn.Parameter(torch.zeros(len(channels)))
        self.frac = S2FracMixBlock(self.c2, self.c2, bins, min_scale, max_scale)
        self.mix = Conv(self.c2 * 2, self.c2, 1, 1)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return x if x.shape[-2:] == ref.shape[-2:] else F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        ref = xs[0]
        feats = [self._resize_like(proj(x), ref) for proj, x in zip(self.proj, xs)]
        weights = self.level_logits.softmax(0).view(-1, 1, 1, 1, 1)
        fused = (torch.stack(feats, 0) * weights).sum(0)
        return self.mix(torch.cat((fused, self.frac(fused)), 1))


class S2FracMixC2f(nn.Module):
    """Complex neck block: C2f split with repeated S2-FracMix feature perturbation."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        expansion: float = 0.5,
        bins: int = 4,
        shortcut: bool = True,
    ):
        super().__init__()
        hidden = max(int(c2 * float(expansion)), 16)
        repeats = max(int(n), 1)
        self.cv1 = Conv(c1, hidden * 2, 1, 1)
        self.blocks = nn.ModuleList(S2FracMixBlock(hidden, hidden, bins) for _ in range(repeats))
        self.cv2 = Conv(hidden * (2 + repeats), c2, 1, 1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for block in self.blocks:
            y.append(block(y[-1]))
        out = self.cv2(torch.cat(y, 1))
        return x + out if self.shortcut else out


class PriorEyeScaleSelect(nn.Module):
    """Head adapter: use PriorEye-style gates to select context from P3/P4/P5."""

    def __init__(self, channels: list[int] | tuple[int, ...], c2: int, prior_channels: int = 16):
        super().__init__()
        self.c2 = int(c2)
        self.proj = nn.ModuleList(Conv(int(c), self.c2, 1, 1) for c in channels)
        self.level = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c2, max(self.c2 // 4, 8), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(self.c2 // 4, 8), len(channels), 1, bias=True),
        )
        self.prior = _VisualSpatialPrior(self.c2, prior_channels)
        self.out = Conv(self.c2, self.c2, 3, 1)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return x if x.shape[-2:] == ref.shape[-2:] else F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        ref = xs[0]
        feats = [self._resize_like(proj(x), ref) for proj, x in zip(self.proj, xs)]
        base = feats[0]
        weights = self.level(base).view(base.shape[0], len(feats), 1, 1, 1).softmax(1)
        mixed = (torch.stack(feats, 1) * weights).sum(1)
        return self.out(base + mixed * self.prior(base))


class PriorEyeDetectAdapter(nn.Module):
    """Detection pre-head adapter combining PriorEye prior gating with S2-FracMix refinement."""

    def __init__(self, c1: int, c2: int, expansion: float = 0.5, bins: int = 4, prior_channels: int = 16):
        super().__init__()
        self.prior = PriorEyeBlock(c1, c2, expansion, prior_channels, 1e-2)
        self.frac = S2FracMixBlock(c2, c2, bins)
        self.score = nn.Sequential(
            nn.Conv2d(c2, max(c2 // 4, 8), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c2 // 4, 8), 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prior = self.prior(x)
        frac = self.frac(prior)
        return self.out(prior + frac * self.score(frac))
