import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import torch.fft
from torch.nn import LayerNorm


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w


        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)

        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)

        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

        print(f'[GlobalLocalFilter] dim={dim}, h={h}, w={w}')

    def forward(self, x):
        x = self.pre_norm(x)

        x1, x2 = torch.chunk(x, 2, dim=1)

        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
        weight = self.complex_weight

        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)
        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


class gnconv(nn.Module):
    def __init__(self, dim, order=3, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dim = dim


        self.base_dims = [dim // (2 ** i) for i in range(order)]
        self.base_dims.reverse()


        self.dims = self.adjust_dims(dim, self.base_dims)

        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)


        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:

            if not callable(gflayer):
                raise TypeError(f"gflayer must be callable, got {type(gflayer).__name__}")


            self.dwconv = gflayer(dim=sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList([
            nn.Conv2d(self.dims[i], self.dims[i + 1], 1)
            for i in range(order - 1)
        ])

        self.scale = s
        print(f'[gnconv] dim={dim}, order={order}, dims={self.dims}, scale={s:.4f}')

    def adjust_dims(self, dim, base_dims):
        """动态调整维度确保分割尺寸匹配"""
        total_needed = 2 * dim
        base_sum = sum(base_dims)
        dims = base_dims.copy()


        adjustment = total_needed - base_dims[0] - base_sum


        dims[-1] += adjustment


        assert base_dims[0] + sum(dims) == 2 * dim, \
            f"维度调整失败: {base_dims[0]} + {sum(dims)} != {2 * dim}"

        return dims

    def forward(self, x):
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = self.proj_out(x)
        return x