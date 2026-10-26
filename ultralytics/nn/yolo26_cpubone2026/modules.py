"""CPUBone adapters for YOLO26.

This module adapts ideas from CPUBone:
"CPUBone: Efficient Vision Backbone Design for Devices with Low
Parallelization Capabilities", CVPR Findings 2026.

Official code: https://github.com/altair199797/CPUBone

The implementation below is detection-oriented rather than a verbatim
classification wrapper.  It keeps the CPUBone CPU-friendly ingredients
-- grouped point-wise convolution, reduced 2x2 kernels in later blocks,
and low-resolution convolutional attention -- while exposing P3/P4/P5
feature maps for Ultralytics-style YOLO YAML files.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


def _to_tuple(value, length: int) -> tuple:
    """Repeat scalar values to a fixed-length tuple."""
    if isinstance(value, (list, tuple)):
        value = list(value)
    else:
        value = [value]
    if len(value) < length:
        value.extend([value[-1]] * (length - len(value)))
    return tuple(value[:length])


def _same_padding(kernel_size: int, stride: int = 1) -> int:
    """CPUBone-style padding; 2x2 stride-1 uses asymmetric padding."""
    if int(kernel_size) == 2:
        return 0 if int(stride) == 2 else -1
    if int(kernel_size) % 2 == 0:
        raise ValueError("Only kernel_size=2 or odd kernels are supported.")
    return int(kernel_size) // 2


def _act(name: str | None) -> nn.Module | None:
    """Build the activation used in CPUBone blocks."""
    if name is None:
        return None
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name in {"hswish", "hardswish"}:
        return nn.Hardswish(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    raise ValueError(f"Unsupported activation: {name}")


class _OpSequential(nn.Module):
    """Sequential that skips None modules."""

    def __init__(self, layers: Sequence[nn.Module | None]):
        super().__init__()
        self.layers = nn.ModuleList([layer for layer in layers if layer is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _Identity(nn.Module):
    """Explicit identity branch used by CPUBone residual wrappers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Residual(nn.Module):
    """Residual wrapper matching the official CPUBone block style."""

    def __init__(self, main: nn.Module | None, shortcut: nn.Module | None):
        super().__init__()
        self.main = main
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            return x
        y = self.main(x)
        return y if self.shortcut is None else y + self.shortcut(x)


class _ConvLayer(nn.Module):
    """Conv + optional BN + optional activation with CPUBone 2x2 padding."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm: bool = True,
        act: str | None = "relu",
    ):
        super().__init__()
        padding = _same_padding(k, s)
        if padding == -1:
            conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(c1, c2, k, s, 0, groups=groups, bias=bias),
            )
        else:
            conv = nn.Conv2d(c1, c2, k, s, padding, groups=groups, bias=bias)
        self.conv = conv
        self.bn = nn.BatchNorm2d(c2) if norm else nn.Identity()
        self.act = _act(act) if act is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _MBConv(nn.Module):
    """CPUBone MBConv with optional grouped expansion projection."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        expand_ratio: float = 4.0,
        grouping: int = 1,
        act: str = "hswish",
    ):
        super().__init__()
        hidden = max(int(round(c1 * float(expand_ratio))), 8)
        grouping = max(1, min(int(grouping), c1, hidden))
        while grouping > 1 and (c1 % grouping != 0 or hidden % grouping != 0):
            grouping -= 1
        self.layers = nn.Sequential(
            _ConvLayer(c1, hidden, 1, 1, groups=grouping, bias=False, norm=False, act=act),
            _ConvLayer(hidden, hidden, k, s, groups=hidden, bias=True, norm=False, act=act),
            _ConvLayer(hidden, c2, 1, 1, bias=False, norm=True, act=None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _FusedMBConv(nn.Module):
    """Fused MBConv branch used by the faster CPUBone variants."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        expand_ratio: float = 4.0,
        grouping: int = 1,
        act: str = "hswish",
    ):
        super().__init__()
        hidden = max(int(round(c1 * float(expand_ratio))), 8)
        grouping = max(1, min(int(grouping), c1, hidden))
        while grouping > 1 and (c1 % grouping != 0 or hidden % grouping != 0):
            grouping -= 1
        self.layers = nn.Sequential(
            _ConvLayer(c1, hidden, k, s, groups=grouping, bias=False, norm=True, act=act),
            _ConvLayer(hidden, c2, 1, 1, bias=False, norm=True, act=None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _ConvAttention(nn.Module):
    """Low-resolution convolutional attention from CPUBone, adapted for feature maps."""

    def __init__(
        self,
        channels: int,
        att_stride: int = 2,
        att_kernel: int = 5,
        head_dim_mul: float = 0.5,
        fuseconv: bool = True,
        smallkernel: bool = True,
        lose_transpose: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = max(1, int((channels * float(head_dim_mul)) // 30))
        self.head_dim = max(1, int((channels // self.num_heads) * float(head_dim_mul)))
        self.num_keys = 3
        self.att_stride = max(int(att_stride), 1)
        kernel = 2 if smallkernel else int(att_kernel)
        total_dim = self.head_dim * self.num_heads * self.num_keys

        self.conv_proj = _ConvLayer(channels, channels, kernel, self.att_stride, groups=channels, norm=True, act=None)
        self.qkv = nn.Conv2d(channels, total_dim, 1, 1, 0, bias=False)
        self.out_channels = self.head_dim * self.num_heads
        self.out_proj = nn.Identity() if fuseconv else nn.Conv2d(self.out_channels, channels, 1, 1, 0)

        if lose_transpose:
            upsample = nn.Upsample(scale_factor=self.att_stride, mode="nearest") if self.att_stride > 1 else nn.Identity()
            self.upsample = nn.Sequential(
                nn.Conv2d(self.out_channels, channels, 1, 1, 0) if fuseconv else nn.Identity(),
                upsample,
            )
        elif self.att_stride == 1:
            self.upsample = nn.ConvTranspose2d(
                self.out_channels if fuseconv else channels,
                channels,
                3,
                1,
                1,
                groups=1 if fuseconv else channels,
            )
        else:
            self.upsample = nn.ConvTranspose2d(
                self.out_channels if fuseconv else channels,
                channels,
                self.att_stride * 2,
                self.att_stride,
                self.att_stride // 2,
                groups=1 if fuseconv else channels,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.conv_proj(x))
        _, _, hh, ww = qkv.shape
        qkv = qkv.reshape(b, self.num_heads, self.num_keys * self.head_dim, hh * ww).permute(0, 1, 3, 2)
        q, k, v = qkv.chunk(3, dim=3)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(max(q.shape[-1], 1))
        y = attn.softmax(dim=-1) @ v
        y = y.permute(0, 1, 3, 2).reshape(b, self.out_channels, hh, ww)
        y = self.upsample(self.out_proj(y))
        return y[:, :c, :h, :w]


class CPUBoneBlock(nn.Module):
    """CPUBone context-plus-local block for later backbone stages."""

    def __init__(
        self,
        channels: int,
        expand_ratio: float = 4.0,
        grouping: int = 2,
        att_stride: int = 1,
        mlp_ratio: int = 2,
        smallkernel: bool = True,
        fuseconv: bool = True,
        lose_transpose: bool = True,
        act: str = "hswish",
    ):
        super().__init__()
        context = _ConvAttention(
            channels,
            att_stride=att_stride,
            att_kernel=5 if att_stride > 1 else 3,
            fuseconv=fuseconv,
            smallkernel=smallkernel,
            lose_transpose=lose_transpose,
        )
        mlp_hidden = max(int(channels * int(mlp_ratio)), 8)
        context_module = nn.Sequential(
            _Residual(nn.Sequential(nn.GroupNorm(1, channels), context), _Identity()),
            _Residual(
                nn.Sequential(
                    nn.GroupNorm(1, channels),
                    nn.Conv2d(channels, mlp_hidden, 1),
                    nn.GELU(approximate="tanh"),
                    nn.Conv2d(mlp_hidden, channels, 1),
                    nn.Dropout(p=0.1),
                ),
                _Identity(),
            ),
        )
        if fuseconv and channels < 256:
            local = _FusedMBConv(channels, channels, 2 if smallkernel else 3, 1, expand_ratio, grouping, act)
        else:
            local = _MBConv(channels, channels, 2 if smallkernel else 3, 1, expand_ratio, grouping, act)
        self.total = nn.Sequential(context_module, _Residual(local, _Identity()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.total(x)


class CPUBoneStableBlock(nn.Module):
    """Layer-scale CPUBone block for stable small-data YOLO fine optimization."""

    def __init__(
        self,
        channels: int,
        expand_ratio: float = 1.5,
        grouping: int = 2,
        att_stride: int = 1,
        layer_scale: float = 0.05,
        use_attention: bool = True,
        act: str = "hswish",
    ):
        super().__init__()
        self.local = _FusedMBConv(
            channels,
            channels,
            k=2,
            s=1,
            expand_ratio=expand_ratio,
            grouping=grouping,
            act=act,
        )
        self.use_attention = bool(use_attention)
        if self.use_attention:
            self.attn_norm = nn.GroupNorm(1, channels)
            self.attn = _ConvAttention(
                channels,
                att_stride=att_stride,
                att_kernel=3,
                fuseconv=True,
                smallkernel=True,
                lose_transpose=True,
            )
        self.gamma_local = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale)))
        self.gamma_attn = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale))) if self.use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.gamma_local * self.local(x)
        if self.use_attention:
            y = y + self.gamma_attn * self.attn(self.attn_norm(y))
        return y


class _CPUBoneStage(nn.Module):
    """One downsampling stage in the CPUBone hierarchy."""

    def __init__(
        self,
        c1: int,
        c2: int,
        depth: int,
        stage_id: int,
        expand_ratio: float = 4.0,
        grouping: int = 2,
        fastit: bool = True,
        bigit: bool = False,
        huge_model: bool = False,
        smallkernel: bool = True,
        lose_transpose: bool = True,
        act: str = "hswish",
    ):
        super().__init__()
        layers = []
        if stage_id <= 2:
            for i in range(max(int(depth), 1)):
                stride = 2 if i == 0 else 1
                ratio = 6 if stride == 2 and (bigit or huge_model) else expand_ratio
                block_cls = _FusedMBConv if fastit else _MBConv
                local = block_cls(c1, c2, 3, stride, ratio, grouping, act)
                layers.append(_Residual(local, _Identity() if stride == 1 and c1 == c2 else None))
                c1 = c2
        else:
            ratio = 6 if bigit or (huge_model and stage_id < 4) else expand_ratio
            first_cls = _FusedMBConv if fastit and (not huge_model or stage_id < 5) else _MBConv
            layers.append(_Residual(first_cls(c1, c2, 3, 2, ratio, grouping, act), None))
            for _ in range(max(int(depth), 1)):
                layers.append(
                    CPUBoneBlock(
                        c2,
                        expand_ratio=expand_ratio,
                        grouping=grouping,
                        att_stride=2 if stage_id == 3 else 1,
                        mlp_ratio=4 if fastit else 2,
                        smallkernel=smallkernel,
                        fuseconv=fastit,
                        lose_transpose=lose_transpose,
                        act=act,
                    )
                )
        self.stage = _OpSequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class CPUBoneBackbone(nn.Module):
    """CPUBone hierarchical backbone returning all internal stages."""

    def __init__(
        self,
        variant: str = "b0",
        in_chans: int = 3,
        expand_ratio: float = 4.0,
        grouping: int = 2,
        smallkernel: bool = True,
        lose_transpose: bool = True,
        fastit: bool = True,
        act: str = "hswish",
    ):
        super().__init__()
        width_list, depth_list = self.variant_settings(variant)
        self.width_list = width_list
        variant = str(variant).lower()
        bigit = variant in {"b1", "b15", "b2", "b25", "b3", "b4", "b5"}
        huge_model = variant in {"b4", "b5"}

        stem_layers = [_ConvLayer(in_chans, width_list[0], 3, 2, norm=True, act=act)]
        for _ in range(max(int(depth_list[0]), 0)):
            stem_layers.append(
                _Residual(
                    _FusedMBConv(width_list[0], width_list[0], 3, 1, 4 if huge_model else 2, grouping, act)
                    if fastit
                    else _MBConv(width_list[0], width_list[0], 3, 1, 4 if huge_model else 2, grouping, act),
                    _Identity(),
                )
            )
        self.stem = _OpSequential(stem_layers)

        c = width_list[0]
        stages = []
        for stage_id, (w, d) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stages.append(
                _CPUBoneStage(
                    c,
                    w,
                    d,
                    stage_id,
                    expand_ratio=expand_ratio,
                    grouping=grouping,
                    fastit=fastit,
                    bigit=bigit,
                    huge_model=huge_model,
                    smallkernel=smallkernel and stage_id >= 3,
                    lose_transpose=lose_transpose,
                    act=act,
                )
            )
            c = w
        self.stages = nn.ModuleList(stages)

    @staticmethod
    def variant_settings(variant: str) -> tuple[list[int], list[int]]:
        """Return official CPUBone width/depth settings."""
        settings = {
            "nano": ([12, 24, 48, 96, 192], [0, 1, 1, 1, 2]),
            "t0": ([12, 24, 48, 96, 192], [0, 1, 1, 1, 3]),
            "s0": ([12, 24, 48, 96, 192], [0, 1, 1, 2, 3]),
            "s1": ([14, 28, 56, 112, 224], [0, 1, 1, 2, 3]),
            "b0": ([16, 32, 64, 128, 256], [0, 1, 1, 3, 4]),
            "b1": ([16, 32, 64, 128, 256], [0, 1, 1, 5, 5]),
            "b15": ([20, 40, 80, 160, 320], [0, 1, 1, 6, 6]),
            "b2": ([24, 48, 96, 192, 384], [0, 1, 1, 6, 6]),
            "b25": ([24, 48, 96, 192, 384], [0, 2, 3, 6, 6]),
            "b3": ([32, 64, 128, 256, 512], [1, 2, 3, 6, 6]),
        }
        return settings.get(str(variant).lower(), settings["b0"])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        x = self.stem(x)  # P1/2
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs


class CPUBoneFeatureAdapter(nn.Module):
    """YOLO channel adapter and light detection refinement for one pyramid level."""

    def __init__(self, c1: int, c2: int, level: int, depth: int = 1, use_context: bool = True):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1)
        level = int(level)
        att_stride = (4, 2, 1)[min(max(level, 0), 2)]
        blocks = []
        if use_context:
            for _ in range(max(int(depth), 0)):
                blocks.append(
                    CPUBoneBlock(
                        c2,
                        expand_ratio=2.0,
                        grouping=2,
                        att_stride=att_stride,
                        mlp_ratio=2,
                        smallkernel=True,
                        fuseconv=True,
                        lose_transpose=True,
                    )
                )
        self.refine = nn.Sequential(*blocks) if blocks else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.proj(x))


class CPUBoneBackboneYOLO(nn.Module):
    """Replace YOLO26 backbone with CPUBone and return adapted P3/P4/P5."""

    def __init__(
        self,
        variant: str = "b0",
        out_channels: Sequence[int] | None = None,
        in_chans: int = 3,
        adapter_depth: int = 1,
        use_context_adapter: bool = True,
    ):
        super().__init__()
        self.backbone = CPUBoneBackbone(
            variant=variant,
            in_chans=in_chans,
            grouping=2,
            smallkernel=True,
            lose_transpose=True,
            fastit=True,
        )
        source_channels = self.backbone.width_list[2:5]
        self.out_channels = list(out_channels) if out_channels is not None else list(source_channels)
        self.adapters = nn.ModuleList(
            CPUBoneFeatureAdapter(c1, int(c2), level=i, depth=adapter_depth, use_context=use_context_adapter)
            for i, (c1, c2) in enumerate(zip(source_channels, self.out_channels))
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        stages = self.backbone(x)
        # stages: P2/4, P3/8, P4/16, P5/32
        pyramid = stages[1:4]
        return [adapter(feat) for adapter, feat in zip(self.adapters, pyramid)]


class CPUBoneP2BackboneYOLO(nn.Module):
    """CPUBone YOLO backbone variant exposing P2/P3/P4/P5 for small objects."""

    def __init__(
        self,
        variant: str = "nano",
        out_channels: Sequence[int] | None = None,
        in_chans: int = 3,
        adapter_depth: int = 0,
        use_context_adapter: bool = False,
    ):
        super().__init__()
        self.backbone = CPUBoneBackbone(
            variant=variant,
            in_chans=in_chans,
            grouping=2,
            smallkernel=True,
            lose_transpose=True,
            fastit=True,
        )
        source_channels = self.backbone.width_list[1:5]
        self.out_channels = list(out_channels) if out_channels is not None else list(source_channels)
        self.adapters = nn.ModuleList(
            CPUBoneFeatureAdapter(c1, int(c2), level=i, depth=adapter_depth, use_context=use_context_adapter)
            for i, (c1, c2) in enumerate(zip(source_channels, self.out_channels))
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        stages = self.backbone(x)
        # stages: P2/4, P3/8, P4/16, P5/32
        pyramid = stages[0:4]
        return [adapter(feat) for adapter, feat in zip(self.adapters, pyramid)]


class CPUBoneFeatureIndex(nn.Module):
    """Select one feature from CPUBoneBackboneYOLO's multi-scale output."""

    def __init__(self, index: int = 0):
        super().__init__()
        self.index = int(index)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return x[self.index]


__all__ = (
    "CPUBoneBackbone",
    "CPUBoneBackboneYOLO",
    "CPUBoneBlock",
    "CPUBoneFeatureAdapter",
    "CPUBoneFeatureIndex",
    "CPUBoneP2BackboneYOLO",
    "CPUBoneStableBlock",
)
