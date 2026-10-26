"""RGB/IR two-stream dynamic fusion blocks for YOLO26.

The modules keep the original YOLO detection head unchanged. RGB and IR are
encoded by paired branches, and interpretable modality weights are generated at
the selected fusion layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C2PSA, C3k2, SPPF
from ultralytics.nn.modules.conv import Conv


class RGBIRSplit(nn.Module):
    """Split input into explicit RGB and IR tensors for dual-backbone YAMLs."""

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[1] >= 4:
            return [x[:, :3], x[:, 3:4]]
        if x.shape[1] == 3:
            return [x, x.mean(1, keepdim=True)]
        if x.shape[1] == 1:
            return [x.repeat(1, 3, 1, 1), x]
        raise ValueError(f"RGBIRSplit expects 1, 3, or 4+ input channels, got {x.shape[1]}")


def _split_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a paired RGB/IR feature tensor into two equal channel groups."""
    if x.ndim != 4:
        raise ValueError(f"RGBIR paired features must be BCHW tensors, got shape {tuple(x.shape)}")
    if x.shape[1] % 2:
        raise ValueError(f"RGBIR paired features require an even channel count, got {x.shape[1]}")
    c = x.shape[1] // 2
    return x[:, :c], x[:, c:]


def _pair_concat(rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
    """Concatenate branch features in RGB-first, IR-second order."""
    return torch.cat((rgb, ir), dim=1)


def _edge_energy(x: torch.Tensor) -> torch.Tensor:
    """Cheap local edge/texture energy used as a differentiable reliability cue."""
    dx = F.pad((x[..., :, 1:] - x[..., :, :-1]).abs().mean(1, keepdim=True), (0, 1, 0, 0))
    dy = F.pad((x[..., 1:, :] - x[..., :-1, :]).abs().mean(1, keepdim=True), (0, 0, 0, 1))
    return dx + dy


def _norm_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a spatial reliability map by its per-image mean."""
    return x / (x.mean(dim=(-2, -1), keepdim=True) + eps)


class RGBIRStem(nn.Module):
    """Split 4-channel RGB/IR input into two modality-specific stems.

    Expected input order is [R, G, B, IR]. If only RGB is provided, an IR proxy is
    built from the RGB mean so smoke tests and RGB-only ablations still run.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        super().__init__()
        self.rgb = Conv(3, c2, k, s)
        self.ir = Conv(1, c2, k, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] >= 4:
            rgb, ir = x[:, :3], x[:, 3:4]
        elif x.shape[1] == 3:
            rgb, ir = x, x.mean(1, keepdim=True)
        elif x.shape[1] == 1:
            ir, rgb = x, x.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"RGBIRStem expects 1, 3, or 4+ input channels, got {x.shape[1]}")
        return _pair_concat(self.rgb(rgb), self.ir(ir))


class RGBIRDualConv(nn.Module):
    """Apply independent Conv blocks to RGB and IR branch features."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=None, g: int = 1, d: int = 1, act=True):
        super().__init__()
        b1 = c1 // 2
        self.rgb = Conv(b1, c2, k, s, p, g, d, act)
        self.ir = Conv(b1, c2, k, s, p, g, d, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, ir = _split_pair(x)
        return _pair_concat(self.rgb(rgb), self.ir(ir))


class RGBIRDualC3k2(nn.Module):
    """Two-stream C3k2 block with branch-specific parameters."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__()
        b1 = c1 // 2
        self.rgb = C3k2(b1, c2, n, c3k, e, attn, g, shortcut)
        self.ir = C3k2(b1, c2, n, c3k, e, attn, g, shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, ir = _split_pair(x)
        return _pair_concat(self.rgb(rgb), self.ir(ir))


class RGBIRDualSPPF(nn.Module):
    """Two-stream SPPF block."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        super().__init__()
        b1 = c1 // 2
        self.rgb = SPPF(b1, c2, k, n, shortcut)
        self.ir = SPPF(b1, c2, k, n, shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, ir = _split_pair(x)
        return _pair_concat(self.rgb(rgb), self.ir(ir))


class RGBIRDualC2PSA(nn.Module):
    """Two-stream C2PSA block."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        b1 = c1 // 2
        self.rgb = C2PSA(b1, c2, n, e)
        self.ir = C2PSA(b1, c2, n, e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, ir = _split_pair(x)
        return _pair_concat(self.rgb(rgb), self.ir(ir))


class RGBIRDynamicFusion(nn.Module):
    """Dynamic RGB/IR fusion with per-frame, per-layer modality weights.

    The gate uses learned local/global logits plus lightweight reliability cues:
    feature texture supports occlusion/detail preservation, while an IR clutter
    prior suppresses large smooth hot regions that often cause false positives.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        reduction: int = 4,
        temperature: float = 1.0,
        prior_strength: float = 0.25,
    ):
        super().__init__()
        b1 = c1 // 2
        hidden = max(c2 // max(int(reduction), 1), 16)
        self.rgb_proj = Conv(b1, c2, 1, 1) if b1 != c2 else nn.Identity()
        self.ir_proj = Conv(b1, c2, 1, 1) if b1 != c2 else nn.Identity()
        gate_channels = c2 * 4
        self.local_gate = nn.Sequential(
            Conv(gate_channels, hidden, 1, 1),
            Conv(hidden, hidden, 3, 1),
            nn.Conv2d(hidden, 2, 1),
        )
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(gate_channels, hidden, 1, 1),
            nn.Conv2d(hidden, 2, 1),
        )
        self.detail = nn.Sequential(Conv(c2 * 3, c2, 1, 1), Conv(c2, c2, 3, 1))
        self.out = Conv(c2 * 2, c2, 1, 1)
        self.temperature = max(float(temperature), 1e-3)
        self.prior_strength = float(prior_strength)
        self.last_weights: torch.Tensor | None = None
        self.last_weight_summary: torch.Tensor | None = None

    def _quality_prior(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        rgb_texture = _norm_map(_edge_energy(rgb) + rgb.float().std(1, keepdim=True))
        ir_texture = _norm_map(_edge_energy(ir) + ir.float().std(1, keepdim=True))
        ir_level = ir.float().abs().mean(1, keepdim=True)
        hot_region = F.relu(ir_level - ir_level.mean(dim=(-2, -1), keepdim=True))
        ir_clutter = _norm_map(hot_region) / (_norm_map(_edge_energy(ir)) + 1.0)
        return torch.cat((rgb_texture, ir_texture - 0.25 * ir_clutter), dim=1).to(dtype=rgb.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, ir = _split_pair(x)
        rgb, ir = self.rgb_proj(rgb), self.ir_proj(ir)
        diff = (rgb - ir).abs()
        prod = rgb * ir
        gate_input = torch.cat((rgb, ir, diff, prod), dim=1)
        logits = self.local_gate(gate_input) + self.global_gate(gate_input)
        logits = logits + self.prior_strength * self._quality_prior(rgb, ir)
        weights = torch.softmax(logits / self.temperature, dim=1)
        self.last_weights = weights.detach()
        self.last_weight_summary = weights.mean(dim=(-2, -1)).detach()

        rgb_w = weights[:, 0:1] * rgb
        ir_w = weights[:, 1:2] * ir
        mixed = rgb_w + ir_w
        detail = self.detail(torch.cat((rgb_w, ir_w, diff), dim=1))
        return self.out(torch.cat((mixed, detail), dim=1))


def collect_rgbir_modal_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect the latest RGB/IR fusion weights from a YOLO model."""
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, RGBIRDynamicFusion) and module.last_weights is not None:
            weights[name] = module.last_weights
    return weights
