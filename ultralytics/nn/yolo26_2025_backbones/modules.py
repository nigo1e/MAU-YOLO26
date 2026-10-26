"""YOLO26-ready adapters inspired by 2025 open-source vision backbones.

The blocks are compact detector-side adaptations rather than full classification
repo copies. They preserve the Ultralytics YAML contract used by C3k2-style
stages: module(c1, c2, n, *args) -> tensor with c2 channels.

References:
- EfficientViM, CVPR 2025: Efficient Vision Mamba with Hidden State Mixer based State Space Duality.
- MobileMamba, 2025: Lightweight Multi-Receptive Visual Mamba Network.
- TinyViM, ICCV 2025: Frequency Decoupling for Tiny Hybrid Vision Mamba.
- OverLoCK, CVPR 2025: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import Conv


def _odd(k: int) -> int:
    """Return an odd kernel size of at least 3."""
    k = max(int(k), 3)
    return k if k % 2 else k + 1


class LayerScale2d(nn.Module):
    """Small learnable residual scale for stable fine-tuning on small data."""

    def __init__(self, channels: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _C2fAdapterStage(nn.Module):
    """C2f-compatible wrapper that keeps cv1/cv2 names for partial weight transfer."""

    block_cls: type[nn.Module]

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, *block_args):
        super().__init__()
        self.c = int(c2 * float(e))
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(self.block_cls(self.c, *block_args) for _ in range(max(int(n), 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class EfficientViMBlock(nn.Module):
    """Hidden-state-mixer-style spatial state update for dense detection features."""

    def __init__(self, channels: int, kernel: int = 5, state_size: int = 4, expansion: float = 1.5):
        super().__init__()
        kernel = _odd(kernel)
        hidden = max(int(channels * float(expansion)), 16)
        self.state_size = max(int(state_size), 1)
        self.norm = nn.BatchNorm2d(channels)
        self.in_proj = Conv(channels, hidden * 2, 1, 1)
        self.local = nn.Conv2d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden, bias=False)
        self.scan_h = nn.Conv2d(hidden, hidden, (1, kernel), padding=(0, kernel // 2), groups=hidden, bias=False)
        self.scan_w = nn.Conv2d(hidden, hidden, (kernel, 1), padding=(kernel // 2, 0), groups=hidden, bias=False)
        self.state_mixer = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=True),
        )
        self.out = Conv(hidden, channels, 1, 1, act=False)
        self.scale = LayerScale2d(channels, 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        u, gate = self.in_proj(x_norm).chunk(2, 1)
        state = F.adaptive_avg_pool2d(u, self.state_size)
        state = self.state_mixer(state)
        state = F.interpolate(state, size=u.shape[-2:], mode="bilinear", align_corners=False)
        y = self.local(u) + self.scan_h(u) + self.scan_w(u) + state
        y = self.out(y * gate.sigmoid())
        return x + self.scale(y)


class MobileMambaBlock(nn.Module):
    """Mobile-friendly multi-receptive gated mixer for feature maps."""

    def __init__(self, channels: int, small_kernel: int = 3, large_kernel: int = 7, expansion: float = 1.25):
        super().__init__()
        small_kernel = _odd(small_kernel)
        large_kernel = _odd(large_kernel)
        hidden = max(int(channels * float(expansion)), 16)
        self.norm = nn.BatchNorm2d(channels)
        self.in_proj = Conv(channels, hidden * 2, 1, 1)
        self.dw_small = nn.Conv2d(hidden, hidden, small_kernel, padding=small_kernel // 2, groups=hidden, bias=False)
        self.dw_large = nn.Conv2d(hidden, hidden, large_kernel, padding=large_kernel // 2, groups=hidden, bias=False)
        self.axial = nn.Sequential(
            nn.Conv2d(hidden, hidden, (1, large_kernel), padding=(0, large_kernel // 2), groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, (large_kernel, 1), padding=(large_kernel // 2, 0), groups=hidden, bias=False),
        )
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, max(hidden // 4, 8), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(hidden // 4, 8), hidden * 3, 1),
        )
        self.out = Conv(hidden, channels, 1, 1, act=False)
        self.scale = LayerScale2d(channels, 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, gate = self.in_proj(self.norm(x)).chunk(2, 1)
        branches = torch.stack((self.dw_small(u), self.dw_large(u), self.axial(u)), 1)
        weights = self.router(u).view(u.shape[0], 3, u.shape[1], 1, 1).softmax(1)
        y = (branches * weights).sum(1)
        return x + self.scale(self.out(y * gate.sigmoid()))


class TinyViMBlock(nn.Module):
    """Frequency-decoupled tiny hybrid mixer for high/low-frequency cues."""

    def __init__(self, channels: int, kernel: int = 5, low_pool: int = 4, expansion: float = 1.0):
        super().__init__()
        kernel = _odd(kernel)
        self.low_pool = max(int(low_pool), 1)
        hidden = max(int(channels * float(expansion)), 16)
        self.norm = nn.BatchNorm2d(channels)
        self.in_proj = Conv(channels, hidden * 2, 1, 1)
        self.high_mix = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.low_h = nn.Conv2d(hidden, hidden, (1, kernel), padding=(0, kernel // 2), groups=hidden, bias=False)
        self.low_w = nn.Conv2d(hidden, hidden, (kernel, 1), padding=(kernel // 2, 0), groups=hidden, bias=False)
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * 2, max(hidden // 4, 8), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(hidden // 4, 8), hidden * 2, 1),
        )
        self.out = Conv(hidden, channels, 1, 1, act=False)
        self.scale = LayerScale2d(channels, 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, gate = self.in_proj(self.norm(x)).chunk(2, 1)
        low = F.avg_pool2d(u, self.low_pool, stride=self.low_pool, ceil_mode=True)
        low = self.low_h(low) + self.low_w(low)
        low = F.interpolate(low, size=u.shape[-2:], mode="bilinear", align_corners=False)
        high = u - F.avg_pool2d(u, 3, stride=1, padding=1)
        high = self.high_mix(high)
        wh, wl = self.freq_gate(torch.cat((high, low), 1)).chunk(2, 1)
        y = high * wh.sigmoid() + low * wl.sigmoid()
        return x + self.scale(self.out(y * gate.sigmoid()))


class OverLoCKBlock(nn.Module):
    """Overview-first and dynamic local-kernel refinement block."""

    def __init__(self, channels: int, kernels: tuple[int, int, int] | list[int] = (3, 5, 9), context_size: int = 4):
        super().__init__()
        k1, k2, k3 = [_odd(k) for k in list(kernels)[:3]]
        self.context_size = max(int(context_size), 1)
        self.norm = nn.BatchNorm2d(channels)
        self.overview = nn.Sequential(
            nn.Conv2d(channels, channels, k3, padding=k3 // 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        self.local_k1 = nn.Conv2d(channels, channels, k1, padding=k1 // 2, groups=channels, bias=False)
        self.local_k2 = nn.Conv2d(channels, channels, k2, padding=k2 // 2, groups=channels, bias=False)
        self.local_k3 = nn.Conv2d(channels, channels, k3, padding=k3 // 2, groups=channels, bias=False)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 8), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), channels * 3, 1),
        )
        self.out = Conv(channels, channels, 1, 1, act=False)
        self.scale = LayerScale2d(channels, 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        overview = F.adaptive_avg_pool2d(x_norm, self.context_size)
        overview = self.overview(overview)
        overview = F.interpolate(overview, size=x_norm.shape[-2:], mode="bilinear", align_corners=False)
        branches = torch.stack((self.local_k1(x_norm), self.local_k2(x_norm), self.local_k3(x_norm)), 1)
        weights = self.router(overview).view(x.shape[0], 3, x.shape[1], 1, 1).softmax(1)
        local = (branches * weights).sum(1)
        return x + self.scale(self.out(local + overview))


class _YOLOBackboneStage(_C2fAdapterStage):
    """C3k2-call-compatible stage whose repeated body is fully replaced."""

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, *block_args):
        del c3k
        super().__init__(c1, c2, n, e, *block_args)


class EfficientViMBackboneStage(_YOLOBackboneStage):
    """Backbone stage replacement using EfficientViM-style blocks."""

    block_cls = EfficientViMBlock


class MobileMambaBackboneStage(_YOLOBackboneStage):
    """Backbone stage replacement using MobileMamba-style blocks."""

    block_cls = MobileMambaBlock


class TinyViMBackboneStage(_YOLOBackboneStage):
    """Backbone stage replacement using TinyViM-style blocks."""

    block_cls = TinyViMBlock


class OverLoCKBackboneStage(_YOLOBackboneStage):
    """Backbone stage replacement using OverLoCK-style blocks."""

    block_cls = OverLoCKBlock


class _C3k2EnhancedStage(C3k2):
    """C3k2-compatible stage with a lightweight module-specific enhancement tail."""

    enhancer_cls: type[nn.Module]

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, attn: bool = False, *args):
        super().__init__(c1, c2, n, c3k=c3k, e=e, attn=attn)
        self.enhance = self.enhancer_cls(c2, *args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enhance(super().forward(x))


class EfficientViMStage(_C3k2EnhancedStage):
    enhancer_cls = EfficientViMBlock


class MobileMambaStage(_C3k2EnhancedStage):
    enhancer_cls = MobileMambaBlock


class TinyViMStage(_C3k2EnhancedStage):
    enhancer_cls = TinyViMBlock


class OverLoCKStage(_C3k2EnhancedStage):
    enhancer_cls = OverLoCKBlock


class VisionBackboneFeatureIndex(nn.Module):
    """Select one feature map from a multi-output full backbone."""

    def __init__(self, index: int = 0):
        super().__init__()
        self.index = int(index)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return x[self.index]


class _FullBackboneYOLO(nn.Module):
    """Detector-friendly full backbone replacement returning [P3, P4, P5]."""

    block_cls: type[nn.Module]
    stage_args: tuple[tuple, ...] = ((), (), (), ())

    def __init__(
        self,
        variant: str = "tiny",
        out_channels: tuple[int, int, int] | list[int] = (256, 512, 1024),
        in_chans: int = 3,
    ):
        super().__init__()
        settings = {
            "nano": ((24, 48, 96, 160, 224), (1, 1, 1, 1)),
            "tiny": ((32, 64, 128, 192, 256), (1, 1, 2, 2)),
            "small": ((48, 96, 160, 256, 384), (1, 2, 2, 2)),
        }
        dims, depths = settings.get(str(variant).lower(), settings["tiny"])
        out_channels = list(out_channels)
        self.out_channels = out_channels

        self.stem = Conv(int(in_chans), dims[0], 3, 2)
        self.p2 = self._make_stage(dims[0], dims[1], depths[0], 0)
        self.p3 = self._make_stage(dims[1], dims[2], depths[1], 1)
        self.p4 = self._make_stage(dims[2], dims[3], depths[2], 2)
        self.p5 = self._make_stage(dims[3], dims[4], depths[3], 3)
        self.adapters = nn.ModuleList(
            [Conv(dims[2], out_channels[0], 1, 1), Conv(dims[3], out_channels[1], 1, 1), Conv(dims[4], out_channels[2], 1, 1)]
        )

    def _make_stage(self, c1: int, c2: int, depth: int, level: int) -> nn.Sequential:
        args = self.stage_args[min(level, len(self.stage_args) - 1)]
        blocks = [Conv(c1, c2, 3, 2)]
        blocks.extend(self.block_cls(c2, *args) for _ in range(max(int(depth), 1)))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [adapter(feat) for adapter, feat in zip(self.adapters, (p3, p4, p5))]


class EfficientViMBackboneYOLO(_FullBackboneYOLO):
    """Full YOLO26 backbone replacement built from EfficientViM-style blocks."""

    block_cls = EfficientViMBlock
    stage_args = ((5, 2, 1.0), (5, 3, 1.0), (7, 4, 1.25), (7, 4, 1.25))


class MobileMambaBackboneYOLO(_FullBackboneYOLO):
    """Full YOLO26 backbone replacement built from MobileMamba-style blocks."""

    block_cls = MobileMambaBlock
    stage_args = ((3, 5, 1.05), (3, 7, 1.10), (3, 9, 1.15), (3, 9, 1.15))


class TinyViMBackboneYOLO(_FullBackboneYOLO):
    """Full YOLO26 backbone replacement built from TinyViM-style blocks."""

    block_cls = TinyViMBlock
    stage_args = ((5, 4, 1.0), (5, 4, 1.0), (7, 4, 1.0), (7, 4, 1.0))


class OverLoCKBackboneYOLO(_FullBackboneYOLO):
    """Full YOLO26 backbone replacement built from OverLoCK-style blocks."""

    block_cls = OverLoCKBlock
    stage_args = (((3, 5, 7), 4), ((3, 5, 9), 4), ((3, 7, 11), 4), ((3, 7, 11), 4))
