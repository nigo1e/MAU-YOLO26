"""MIT 2025 LinOSS/D-LinOSS inspired modules for YOLO26.

These adapters translate oscillatory state-space dynamics into 2D detection
features.  Each block keeps the Ultralytics YAML contract:
    module(c1, c2, *args) -> Tensor[B, c2, H, W]
so the detector head, losses, trainer, and existing improvement packages remain
unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv


def _oscillatory_step(x: torch.Tensor, velocity: torch.Tensor, omega: torch.Tensor, damping: torch.Tensor):
    """One stable damped oscillator update for feature tensors."""

    force = -omega.square() * x - damping * velocity
    velocity = velocity + force
    state = x + velocity
    return state, velocity


class DLinOSSBackboneBlock(nn.Module):
    """Damped LinOSS residual block for YOLO26 backbone feature refinement."""

    def __init__(self, c1: int, c2: int, expansion: float = 0.5, steps: int = 2, damping_max: float = 0.9):
        super().__init__()
        hidden = max(int(c2 * expansion), 8)
        self.steps = max(int(steps), 1)
        self.damping_max = float(damping_max)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.reduce = Conv(c2, hidden, 1, 1)
        self.spatial = DWConv(hidden, hidden, 3, 1)
        self.omega = nn.Sequential(nn.Conv2d(hidden, hidden, 1, bias=True), nn.Sigmoid())
        self.damping = nn.Sequential(nn.Conv2d(hidden, hidden, 1, bias=True), nn.Sigmoid())
        self.expand = Conv(hidden, c2, 1, 1, act=False)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, c2, 1, bias=True), nn.Sigmoid())
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        state = self.spatial(self.reduce(x))
        velocity = torch.zeros_like(state)
        omega = 0.1 + 1.9 * self.omega(state)
        damping = self.damping_max * self.damping(state)
        for _ in range(self.steps):
            state, velocity = _oscillatory_step(state, velocity, omega, damping)
        update = self.expand(state)
        return self.out(x + self.gate(x) * update)


class LinOSSNeckFusion(nn.Module):
    """Oscillatory multi-scale fusion block for YOLO26 neck concatenations."""

    def __init__(self, c1: int, c2: int, branches: int = 3, expansion: float = 0.5, steps: int = 2):
        super().__init__()
        hidden = max(int(c2 * expansion), 8)
        self.steps = max(int(steps), 1)
        self.proj = Conv(c1, c2, 1, 1)
        self.branches = nn.ModuleList(DWConv(c2, c2, 3 + 2 * i, 1) for i in range(max(int(branches), 1)))
        self.context = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, hidden, 1, bias=True), nn.SiLU(inplace=True))
        self.mix = nn.Conv2d(hidden, len(self.branches), 1, bias=True)
        self.omega = nn.Sequential(nn.Conv2d(hidden, c2, 1, bias=True), nn.Sigmoid())
        self.damping = nn.Sequential(nn.Conv2d(hidden, c2, 1, bias=True), nn.Sigmoid())
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        context = self.context(x)
        weights = self.mix(context).softmax(1)
        feats = torch.stack([branch(x) for branch in self.branches], dim=1)
        target = (feats * weights.unsqueeze(2)).sum(1)
        state = x
        velocity = target - x
        omega = 0.1 + 1.9 * self.omega(context)
        damping = 0.9 * self.damping(context)
        for _ in range(self.steps):
            residual = state - target
            force = -omega.square() * residual - damping * velocity
            velocity = velocity + force
            state = state + velocity
        return self.out(state)


class AxialLinOSSScan(nn.Module):
    """Bidirectional axial LinOSS scan for long-range pre-detection context."""

    def __init__(self, c1: int, c2: int, expansion: float = 0.5, steps: int = 2):
        super().__init__()
        hidden = max(int(c2 * expansion), 8)
        self.steps = max(int(steps), 1)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.reduce = Conv(c2, hidden, 1, 1)
        self.row_mixer = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.col_mixer = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.params = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden, hidden * 2, 1, bias=True), nn.Sigmoid())
        self.expand = Conv(hidden, c2, 1, 1, act=False)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, c2, 1, bias=True), nn.Sigmoid())
        self.out = Conv(c2, c2, 1, 1)

    def _scan_rows(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = x.permute(0, 2, 1, 3).reshape(b * h, c, w)
        y = self.row_mixer(y) + torch.flip(self.row_mixer(torch.flip(y, dims=(-1,))), dims=(-1,))
        return y.reshape(b, h, c, w).permute(0, 2, 1, 3)

    def _scan_cols(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = x.permute(0, 3, 1, 2).reshape(b * w, c, h)
        y = self.col_mixer(y) + torch.flip(self.col_mixer(torch.flip(y, dims=(-1,))), dims=(-1,))
        return y.reshape(b, w, c, h).permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        state = self.reduce(x)
        omega, damping = self.params(state).chunk(2, dim=1)
        omega = 0.1 + 1.9 * omega
        damping = 0.9 * damping
        velocity = torch.zeros_like(state)
        for _ in range(self.steps):
            target = self._scan_rows(state) + self._scan_cols(state)
            force = -omega.square() * (state - target) - damping * velocity
            velocity = velocity + force
            state = state + velocity
        update = self.expand(state)
        return self.out(x + self.gate(x) * update)
