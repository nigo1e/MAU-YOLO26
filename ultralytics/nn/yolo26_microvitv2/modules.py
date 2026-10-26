"""MicroViTv2 adapters for YOLO26.

The code in this file is adapted from the official MicroViT repository:
https://github.com/novendrastywn/MicroViT

It keeps Ultralytics YAML compatibility by exposing modules whose constructor
signatures match ``module(c1, c2, *args)`` when they are used as ordinary YOLO
blocks, and by exposing dedicated backbone wrappers for multi-output features.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


def _to_list(value, n: int):
    """Return a list with length n."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return [value for _ in range(n)]


class Conv2dBN(nn.Sequential):
    """Conv2d + BatchNorm2d block used by MicroViTv2."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        d: int = 1,
        g: int = 1,
        bn_weight_init: float = 1.0,
    ):
        p = k // 2 if p is None else p
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, groups=g, bias=False),
            nn.BatchNorm2d(c2),
        )
        nn.init.constant_(self[1].weight, bn_weight_init)
        nn.init.constant_(self[1].bias, 0)


class MicroRepConv(nn.Module):
    """MicroViTv2 RepConv branch before deployment re-parameterization."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None, g: int = 1):
        super().__init__()
        p = k // 2 if p is None else p
        rep_k = max(k // 2, 1)
        rep_p = rep_k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=True)
        self.repconv = nn.Conv2d(c1, c2, rep_k, s, rep_p, groups=g, bias=True)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x) + self.repconv(x))


class MicroResidual(nn.Module):
    """Residual wrapper used by MicroViTv2."""

    def __init__(self, module: nn.Module, drop: float = 0.0):
        super().__init__()
        self.module = module
        self.drop = float(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop > 0:
            keep = x.new_empty(x.size(0), 1, 1, 1).bernoulli_(1.0 - self.drop).div_(1.0 - self.drop)
            return x + self.module(x) * keep
        return x + self.module(x)


class MicroFFN(nn.Module):
    """MicroViTv2 point-wise feed-forward network for 2D features."""

    def __init__(self, channels: int, hidden: int, act_layer: type[nn.Module] = nn.SiLU):
        super().__init__()
        self.net = nn.Sequential(
            Conv2dBN(channels, hidden, 1, 1, 0),
            act_layer(),
            Conv2dBN(hidden, channels, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MicroESHA(nn.Module):
    """Efficient self-hybrid attention from MicroViTv2."""

    def __init__(
        self,
        channels: int,
        partial_ratio: float = 0.25,
        qk_dim: int = 16,
        inp_group: int = 32,
        act_layer: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.qk_dim = int(qk_dim)
        self.channels = int(channels)
        self.partial_dim = max(int(channels * partial_ratio), 1)
        self.split_index = (self.qk_dim, self.qk_dim, self.partial_dim, channels - self.partial_dim)
        groups = max(1, min(inp_group, channels))
        hidden = self.qk_dim * 2 + channels
        while groups > 1 and (channels % groups != 0 or hidden % groups != 0):
            groups -= 1
        self.scale = self.qk_dim**-0.5
        self.pre_norm = nn.GroupNorm(1, channels)
        self.in_proj = MicroRepConv(channels, hidden, 3, 1, 1, groups)
        self.out_proj = nn.Sequential(act_layer(), Conv2dBN(channels, channels, 1, 1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        q, k, v, u = self.in_proj(x).split(self.split_index, dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        b, _, h, w = u.shape
        attended = (v @ attn.transpose(-2, -1)).reshape(b, self.partial_dim, h, w)
        return self.out_proj(torch.cat((attended, u), dim=1))


class MicroSDTA(nn.Module):
    """Spatial-depthwise token attention from MicroViTv2."""

    def __init__(
        self,
        channels: int,
        partial_ratio: float = 0.25,
        qk_dim: int = 16,
        act_layer: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.qk_dim = int(qk_dim)
        self.channels = int(channels)
        self.partial_dim = max(int(channels * partial_ratio), 1)
        self.split_index = (self.qk_dim, self.qk_dim, self.partial_dim, channels - self.partial_dim)
        self.scale = self.qk_dim**-0.5
        self.pre_norm = nn.GroupNorm(1, channels)
        hidden = self.qk_dim * 2 + channels
        self.in_proj = nn.Sequential(
            MicroRepConv(channels, channels, 3, 1, 1, channels),
            Conv2dBN(channels, hidden, 1, 1, 0),
        )
        self.out_proj = nn.Sequential(act_layer(), Conv2dBN(channels, channels, 1, 1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        q, k, v, u = self.in_proj(x).split(self.split_index, dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        b, _, h, w = u.shape
        attended = (v @ attn.transpose(-2, -1)).reshape(b, self.partial_dim, h, w)
        return self.out_proj(torch.cat((attended, u), dim=1))


class MicroViTv2Block(nn.Module):
    """A single MicroViTv2 block adapted to YOLO feature maps."""

    def __init__(
        self,
        c1: int,
        c2: int,
        block_type: str = "c",
        mlp_ratio: float = 2.0,
        attn_ratio: float = 0.25,
        qk_dim: int = 16,
        shortcut: bool = True,
        act: str = "silu",
    ):
        super().__init__()
        act_layer = nn.SiLU if str(act).lower() == "silu" else nn.ReLU
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        block_type = str(block_type).lower()
        aliases = {"fast": "f", "full": "f", "rep": "f", "conv": "c", "attention": "esha"}
        block_type = aliases.get(block_type, block_type)
        hidden = max(int(c2 * mlp_ratio), 16)
        if block_type == "f":
            body = MicroResidual(
                nn.Sequential(
                    MicroRepConv(c2, c2, 3, 1, 1, 1),
                    act_layer(),
                    Conv2dBN(c2, c2, 1, 1, 0),
                )
            )
            self.block = body
        else:
            if block_type == "c":
                spatial_mix = MicroRepConv(c2, c2, 3, 1, 1, c2)
            elif block_type == "esha":
                spatial_mix = MicroESHA(c2, attn_ratio, qk_dim, act_layer=nn.GELU)
            elif block_type == "sdta":
                spatial_mix = MicroSDTA(c2, attn_ratio, qk_dim, act_layer=nn.GELU)
            else:
                raise ValueError(f"Unsupported MicroViTv2 block_type={block_type!r}")
            self.block = nn.Sequential(
                MicroResidual(spatial_mix),
                MicroResidual(MicroFFN(c2, hidden, act_layer)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.proj(x))


class MicroViTv2Stage(nn.Module):
    """Stack of MicroViTv2 blocks for direct insertion in a YOLO backbone."""

    def __init__(
        self,
        c1: int,
        c2: int,
        depth: int = 2,
        block_type: str = "c",
        mlp_ratio: float = 2.0,
        attn_ratio: float = 0.25,
        qk_dim: int = 16,
    ):
        super().__init__()
        depth = max(int(depth), 1)
        blocks = [MicroViTv2Block(c1, c2, block_type, mlp_ratio, attn_ratio, qk_dim, shortcut=False)]
        blocks.extend(
            MicroViTv2Block(c2, c2, block_type, mlp_ratio, attn_ratio, qk_dim, shortcut=True)
            for _ in range(depth - 1)
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MicroViTv2Stem(nn.Module):
    """Detection-friendly MicroViTv2 stem.

    The official classification/downstream stem can start at patch_size 16. For
    YOLO detection we start the hierarchical trunk at /4, then stages 1/2/3
    become P3/8, P4/16 and P5/32.
    """

    def __init__(self, c1: int, c2: int, act_layer: type[nn.Module] = nn.SiLU):
        super().__init__()
        mid = max(c2 // 2, 16)
        self.stem = nn.Sequential(
            MicroRepConv(c1, mid, 3, 2, 1),
            act_layer(),
            MicroRepConv(mid, c2, 3, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class MicroViTv2PatchMerging(nn.Module):
    """MicroViTv2 downsampling projection."""

    def __init__(self, c1: int, c2: int, act_layer: type[nn.Module] = nn.SiLU):
        super().__init__()
        self.down = nn.Sequential(
            MicroResidual(MicroRepConv(c1, c1, 3, 1, 1, c1)),
            MicroResidual(MicroFFN(c1, c1 * 2, act_layer)),
            MicroRepConv(c1, c2, 3, 2, 1),
            MicroResidual(MicroRepConv(c2, c2, 3, 1, 1, c2)),
            MicroResidual(MicroFFN(c2, c2 * 2, act_layer)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class MicroFeatureIndex(nn.Module):
    """Select one feature from a MicroViTv2 multi-scale output list."""

    def __init__(self, index: int = 0):
        super().__init__()
        self.index = int(index)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return x[self.index]


class MicroViTv2Backbone(nn.Module):
    """Multi-output MicroViTv2 backbone producing YOLO P3/P4/P5 features."""

    def __init__(
        self,
        in_chans: int = 3,
        dims: Sequence[int] = (64, 128, 256, 512),
        depths: Sequence[int] = (1, 2, 4, 3),
        types: Sequence[str] = ("f", "c", "c", "sdta"),
        attn_ratios: Sequence[float] = (0.0, 0.0, 0.0, 0.25),
        mlp_ratio: float = 2.0,
        qk_dim: int = 16,
        out_indices: Sequence[int] = (1, 2, 3),
        act: str = "silu",
    ):
        super().__init__()
        dims = list(dims)
        depths = list(depths)
        num_stages = len(dims)
        types = _to_list(types, num_stages)
        attn_ratios = _to_list(attn_ratios, num_stages)
        act_layer = nn.SiLU if str(act).lower() == "silu" else nn.ReLU
        self.out_indices = tuple(int(i) for i in out_indices)
        self.out_channels = [dims[i] for i in self.out_indices]

        self.stem = MicroViTv2Stem(in_chans, dims[0], act_layer)
        stages = []
        downs = []
        for i in range(num_stages):
            stages.append(
                MicroViTv2Stage(
                    dims[i],
                    dims[i],
                    depths[i],
                    types[i],
                    mlp_ratio,
                    float(attn_ratios[i]),
                    qk_dim,
                )
            )
            if i < num_stages - 1:
                downs.append(MicroViTv2PatchMerging(dims[i], dims[i + 1], act_layer))
        self.stages = nn.ModuleList(stages)
        self.downs = nn.ModuleList(downs)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return outs


class MicroViTv2BackboneYOLO(nn.Module):
    """Method 1: replace the YOLO backbone with a MicroViTv2 multi-scale trunk."""

    def __init__(
        self,
        variant: str = "tiny",
        out_channels: Sequence[int] | None = None,
        in_chans: int = 3,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        settings = {
            "tiny": dict(dims=(64, 128, 256, 512), depths=(1, 2, 3, 2), types=("f", "c", "c", "sdta")),
            "small": dict(dims=(96, 192, 320, 512), depths=(1, 2, 4, 3), types=("f", "c", "c", "sdta")),
            "official3": dict(dims=(96, 192, 384, 448), depths=(1, 3, 7, 6), types=("f", "c", "c", "sdta")),
            "microvitv2_3": dict(dims=(96, 192, 384, 448), depths=(1, 3, 7, 6), types=("f", "c", "c", "sdta")),
        }
        cfg = settings.get(str(variant).lower(), settings["tiny"])
        self.backbone = MicroViTv2Backbone(in_chans=in_chans, mlp_ratio=mlp_ratio, **cfg)
        channels = list(self.backbone.out_channels)
        if out_channels is None:
            self.out_channels = channels
            self.proj = nn.ModuleList(nn.Identity() for _ in channels)
        else:
            out_channels = list(out_channels)
            self.out_channels = out_channels
            self.proj = nn.ModuleList(Conv(c1, c2, 1, 1) for c1, c2 in zip(channels, out_channels))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [proj(feat) for proj, feat in zip(self.proj, self.backbone(x))]


class MicroViTv2AdapterYOLO(nn.Module):
    """Method 2: MicroViTv2 backbone plus explicit YOLO channel adapters."""

    def __init__(
        self,
        variant: str = "tiny",
        yolo_channels: Sequence[int] = (256, 512, 1024),
        in_chans: int = 3,
        mlp_ratio: float = 2.0,
        adapter_depth: int = 1,
    ):
        super().__init__()
        self.trunk = MicroViTv2BackboneYOLO(variant, out_channels=None, in_chans=in_chans, mlp_ratio=mlp_ratio)
        self.out_channels = list(yolo_channels)
        self.adapters = nn.ModuleList()
        for c1, c2 in zip(self.trunk.out_channels, self.out_channels):
            layers = [Conv(c1, c2, 1, 1)]
            for _ in range(max(int(adapter_depth), 0)):
                layers.append(MicroViTv2Block(c2, c2, "c", mlp_ratio=1.5, attn_ratio=0.0, shortcut=True))
            self.adapters.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.trunk(x)
        return [adapter(feat) for adapter, feat in zip(self.adapters, feats)]
