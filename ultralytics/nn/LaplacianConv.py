import torch
import torch.nn as nn
import torch.nn.functional as F
class LaplacianConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None,
                 groups=1, act=True, alpha_init=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=(kernel_size // 2) if padding is None else padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()
        laplacian_kernel = torch.tensor([[-1, -1, -1],
                                         [-1,  8, -1],
                                         [-1, -1, -1]], dtype=torch.float32)
        self.register_buffer('laplacian_kernel', laplacian_kernel.view(1, 1, 3, 3))
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
    def forward(self, x):

        b, c, h, w = x.shape

        kernel = self.laplacian_kernel.expand(c, 1, 3, 3)   # [c, 1, 3, 3]
        edge = F.conv2d(x, kernel, padding=1, groups=c)    # [b, c, h, w]
        x_enhanced = x + self.alpha * edge
        return self.act(self.bn(self.conv(x_enhanced)))
