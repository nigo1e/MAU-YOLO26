"""YOLO26-ready adapters inspired by 2026 research modules.

These are detector-friendly re-interpretations that keep the Ultralytics YAML
contract:
    module(c1, c2, *args) -> feature map with c2 channels
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv


def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Build a normalized 2D Gaussian kernel."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x.square() + grid_y.square()) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


class VECABlock(nn.Module):
    """Core-periphery attention block distilled from VECA."""

    def __init__(self, c1: int, c2: int, cores: int = 16, heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.cores = max(int(cores), 4)
        self.heads = max(math.gcd(int(heads), c2), 1)
        hidden = max(int(c2 * mlp_ratio), 16)

        self.core_tokens = nn.Parameter(torch.randn(1, self.cores, c2) * 0.02)
        self.patch_norm = nn.LayerNorm(c2)
        self.core_norm = nn.LayerNorm(c2)
        self.core_attn = nn.MultiheadAttention(c2, self.heads, batch_first=True)
        self.patch_attn = nn.MultiheadAttention(c2, self.heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(c2),
            nn.Linear(c2, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, c2),
        )
        self.local = DWConv(c2, c2, 3, 1)
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, c, h, w = x.shape
        patches = x.flatten(2).transpose(1, 2)
        cores = self.core_tokens.expand(b, -1, -1)

        merged = torch.cat((cores, patches), 1)
        core_delta = self.core_attn(
            self.core_norm(cores),
            self.patch_norm(merged),
            self.patch_norm(merged),
            need_weights=False,
        )[0]
        cores = cores + core_delta

        patch_delta = self.patch_attn(
            self.patch_norm(patches),
            self.core_norm(cores),
            self.core_norm(cores),
            need_weights=False,
        )[0]
        patches = patches + patch_delta
        patches = patches + self.mlp(patches)

        y = patches.transpose(1, 2).reshape(b, c, h, w)
        return self.out(x + self.local(y))


class DRoRAEBlock(nn.Module):
    """Depth-routed fusion block distilled from DRoRAE."""

    def __init__(self, c1: int, c2: int, experts: int = 4, expansion: float = 0.5, correction_scale: float = 0.25):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        hidden = max(int(c2 * expansion), 16)

        kernels = [1, 3, 5, 7][: max(int(experts), 2)]
        self.experts = nn.ModuleList()
        for k in kernels:
            if k == 1:
                self.experts.append(
                    nn.Sequential(
                        Conv(c2, hidden, 1, 1),
                        Conv(hidden, c2, 1, 1, act=False),
                    )
                )
            else:
                self.experts.append(
                    nn.Sequential(
                        Conv(c2, hidden, 1, 1),
                        DWConv(hidden, hidden, k, 1),
                        Conv(hidden, c2, 1, 1, act=False),
                    )
                )

        self.router = nn.Conv2d(c2, len(self.experts), 1, bias=True)
        self.mix = Conv(c2, c2, 1, 1)
        self.scale = nn.Parameter(torch.tensor(float(correction_scale)))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        expert_feats = torch.stack([expert(x) for expert in self.experts], 1)
        weights = torch.tanh(self.router(x))
        weights = weights / (weights.abs().sum(1, keepdim=True) + 1e-6)
        fused = (expert_feats * weights.unsqueeze(2)).sum(1)
        correction = self.mix(fused)
        return self.act(x + self.scale * correction)


class MVSplitBlock(nn.Module):
    """Mean-variance split residual block distilled from MV-Split."""

    def __init__(self, c1: int, c2: int, expansion: float = 2.0, leak: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        hidden = max(int(c2 * expansion), 16)
        self.residual = nn.Sequential(
            Conv(c2, hidden, 1, 1),
            DWConv(hidden, hidden, 3, 1),
            nn.Conv2d(hidden, c2, 1, bias=False),
        )
        self.center_scale = nn.Parameter(torch.ones(1, c2, 1, 1))
        self.mean_scale = nn.Parameter(torch.full((1, c2, 1, 1), float(leak)))
        self.out = Conv(c2, c2, 1, 1)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        resid = self.residual(x)
        resid_mean = resid.mean((2, 3), keepdim=True)
        centered = resid - resid_mean
        centered = centered * torch.rsqrt(centered.square().mean((2, 3), keepdim=True) + self.eps)

        trunk_mean = x.mean((2, 3), keepdim=True)
        mean_update = self.mean_scale * (resid_mean - trunk_mean)
        return self.out(x + self.center_scale * centered + mean_update)


class UpsampleAnything(nn.Module):
    """Edge-aware feature upsampler distilled from Upsample Anything."""

    def __init__(self, c1: int, c2: int, scale: int = 2, kernel_size: int = 5, sigma: float = 1.2):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.scale = max(int(scale), 2)
        kernel_size = max(int(kernel_size), 3)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.smooth = nn.Conv2d(c2, c2, kernel_size, padding=kernel_size // 2, groups=c2, bias=False)
        self.detail = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
        self.out = Conv(c2, c2, 1, 1)

        kernel = _gaussian_kernel(kernel_size, float(sigma))
        with torch.no_grad():
            self.smooth.weight.copy_(kernel.view(1, 1, kernel_size, kernel_size).repeat(c2, 1, 1, 1))

    @staticmethod
    def _edge_map(x: torch.Tensor) -> torch.Tensor:
        gx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        gy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        up = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        guide = F.interpolate(x.mean(1, keepdim=True), scale_factor=self.scale, mode="bilinear", align_corners=False)
        edge = self.edge_gate(self._edge_map(guide))
        smooth = self.smooth(up)
        detail = self.detail(up - smooth)
        refined = up + edge * detail + 0.5 * (1.0 - edge) * (smooth - up)
        return self.out(refined)


class XRestormerPPBlock(nn.Module):
    """All-weather refinement block distilled from X-Restormer++."""

    def __init__(self, c1: int, c2: int, expansion: float = 2.0, reduction: int = 4):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        hidden = max(int(c2 * expansion), 16)
        mid = max(c2 // max(int(reduction), 1), 8)

        self.scale_map = nn.Sequential(
            nn.Conv2d(c2, mid, 3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 1, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.local = nn.Sequential(
            DWConv(c2, c2, 3, 1),
            DWConv(c2, c2, 5, 1),
        )
        self.global_mixer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.edge_proj = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = Conv(c2 * 2, c2, 1, 1)
        self.out = Conv(c2, c2, 1, 1)

    @staticmethod
    def _edge_map(x: torch.Tensor) -> torch.Tensor:
        gx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        gy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        return torch.sqrt(gx.square() + gy.square() + 1e-6).mean(1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        scaled = x * (1.0 + self.scale_map(x))
        local = self.local(scaled)
        global_ctx = scaled * self.global_mixer(scaled)
        edge = self.edge_proj(self._edge_map(scaled))
        refined = self.fuse(torch.cat((local + edge * scaled, global_ctx), 1))
        return self.out(scaled + refined)
