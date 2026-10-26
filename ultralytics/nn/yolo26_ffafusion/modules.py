

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect


def _odd_kernel(k: int) -> int:
    """Return a positive odd kernel size."""
    k = max(int(k), 3)
    return k if k % 2 else k + 1


def _meshgrid(y: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for torch.meshgrid."""
    try:
        return torch.meshgrid(y, x, indexing="ij")
    except TypeError:
        return torch.meshgrid(y, x)


def _wrap_pi(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angle differences to [-pi, pi]."""
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def _estimate_fourier_angle(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Estimate the dominant feature orientation from the Fourier spectrum.

    Args:
        x: Feature map with shape [B, C, H, W].
        eps: Numerical stability constant.

    Returns:
        Tensor with shape [B], containing one dominant orientation per image.
    """
    b, _, h, w = x.shape
    if h < 3 or w < 3:
        return x.new_zeros(b)

    # Use activation energy so opposite-sign channels do not cancel each other.
    energy_map = x.detach().float().pow(2).mean(1)
    energy_map = energy_map - energy_map.mean(dim=(-2, -1), keepdim=True)

    spectrum = torch.fft.fftshift(torch.fft.fft2(energy_map, norm="ortho"), dim=(-2, -1)).abs()
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, d=1.0)).to(x.device) * h
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, d=1.0)).to(x.device) * w
    yy, xx = _meshgrid(fy, fx)
    rho = torch.sqrt(xx.square() + yy.square())
    theta = torch.atan2(yy, xx)
    mask = rho > eps

    weights = spectrum * rho.unsqueeze(0)
    weights = weights[:, mask]
    theta = theta[mask]
    if weights.numel() == 0:
        return x.new_zeros(b)

    # Axial orientation: theta and theta + pi are equivalent, hence 2 * theta.
    cos2 = (weights * torch.cos(2.0 * theta).unsqueeze(0)).sum(1)
    sin2 = (weights * torch.sin(2.0 * theta).unsqueeze(0)).sum(1)
    return 0.5 * torch.atan2(sin2, cos2).to(dtype=x.dtype)


def _rotate_feature(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate each feature map in a batch by its own angle."""
    b, _, _, _ = x.shape
    theta = theta.to(device=x.device, dtype=x.dtype)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    affine = x.new_zeros(b, 2, 3)
    affine[:, 0, 0] = cos_t
    affine[:, 0, 1] = -sin_t
    affine[:, 1, 0] = sin_t
    affine[:, 1, 1] = cos_t
    grid = F.affine_grid(affine, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


class FourierAngleAlign(nn.Module):
    """Estimate a Fourier angle and align a feature map to a reference/canonical angle."""

    def __init__(self, channels: int, m: int = 7, c_mid: int = 32, layer_scale_init_value: float = 1e-3):
        super().__init__()
        m = _odd_kernel(m)
        hidden = max(min(int(c_mid), channels), 8)
        self.local = nn.Sequential(
            Conv(channels, hidden, 1, 1),
            DWConv(hidden, hidden, m, 1),
            Conv(hidden, channels, 1, 1, act=False),
        )
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale_init_value)))

    def forward(self, x: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
        theta_x = _estimate_fourier_angle(x)
        theta_ref = x.new_zeros(theta_x.shape) if reference is None else _estimate_fourier_angle(reference)
        delta = _wrap_pi(theta_ref - theta_x)
        aligned = _rotate_feature(x, delta)
        return x + self.gamma * self.local(aligned - x)


class FFAFusionBlock(nn.Module):
    """Backbone/head-side Fourier angle alignment refinement block."""

    def __init__(
        self,
        c1: int,
        c2: int,
        m: int = 7,
        c_mid: int = 32,
        layer_scale_init_value: float = 1e-3,
        expansion: float = 0.5,
    ):
        super().__init__()
        hidden = max(int(c2 * float(expansion)), 16)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.align = FourierAngleAlign(c2, m, c_mid, layer_scale_init_value)
        self.refine = nn.Sequential(Conv(c2, hidden, 1, 1), DWConv(hidden, hidden, 3, 1), Conv(hidden, c2, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.refine(self.align(x))


class FFAFusionConcat(nn.Module):
    """Neck FAAFusion adapter for YOLO PAN-FPN concat nodes.

    The last input feature is treated as the direction reference.  Previous
    inputs are rotated according to Fourier-estimated direction differences,
    then concatenated with the reference.  This keeps the same output channel
    count as a normal Concat, so downstream YOLO26 layers remain unchanged.
    """

    def __init__(
        self,
        channels: list[int],
        dimension: int = 1,
        m: int = 7,
        c_mid: int = 32,
        layer_scale_init_value: float = 1e-3,
    ):
        super().__init__()
        if dimension != 1:
            raise ValueError("FFAFusionConcat only supports channel-dimension fusion.")
        self.d = dimension
        self.align = nn.ModuleList(
            FourierAngleAlign(ch, m, c_mid, layer_scale_init_value) for ch in channels[:-1]
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        reference = x[-1]
        target_size = reference.shape[-2:]
        aligned = []
        for feat, align in zip(x[:-1], self.align):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="nearest")
            aligned.append(align(feat, reference))
        return torch.cat([*aligned, reference], self.d)


class FFAFusionDetect(Detect):
    """YOLO Detect head with Fourier angle pre-alignment before prediction."""

    def __init__(
        self,
        nc: int = 80,
        m: int = 7,
        c_mid: int = 32,
        layer_scale_init_value: float = 1e-3,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        super().__init__(nc, reg_max, end2end, ch)
        self.ffa = nn.ModuleList(FFAFusionBlock(c, c, m, c_mid, layer_scale_init_value, 0.5) for c in ch)
        if end2end:
            self.one2one_ffa = copy.deepcopy(self.ffa)

    @property
    def one2many(self):
        """Return one-to-many prediction branches."""
        return dict(box_head=self.cv2, cls_head=self.cv3, ffa=self.ffa)

    @property
    def one2one(self):
        """Return one-to-one prediction branches for end-to-end mode."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, ffa=self.one2one_ffa)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module = None,
        cls_head: torch.nn.Module = None,
        ffa: torch.nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None or ffa is None:
            return dict()
        refined = [ffa[i](x[i]) for i in range(self.nl)]
        return super().forward_head(refined, box_head, cls_head)

    def fuse(self) -> None:
        """Remove one-to-many branches during inference optimization."""
        self.cv2 = self.cv3 = self.ffa = None
