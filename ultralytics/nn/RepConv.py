import torch
import torch.nn as nn
from torch.nn import functional as F


class RepConv(nn.Module):
    """
    实现重参数化卷积（RepConv），通过对传统卷积核的动态调整来提升表示能力。

    参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - kernel_size (int): 卷积核大小。
    - stride (int): 卷积步长。
    - padding (int, optional): 卷积填充大小，默认根据kernel_size自动计算。
    - groups (int): 分组卷积中的组数，默认为1，表示不使用分组卷积。
    - map_k (int): 用于动态调整卷积核的小卷积核大小，默认为3。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1, map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size


        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)

        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))


        G = in_channels * out_channels // (groups ** 2)

        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size


        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)

        self.bias = None

        self.stride = stride
        self.groups = groups


        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        """
        前向传播函数。

        参数:
        - inputs (Tensor): 输入特征图。

        返回:
        - 经过重参数化卷积操作后的输出特征图。
        """

        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)

        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)

        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups,
                        bias=self.bias)



if __name__ == '__main__':
    block = RepConv(64, 64, kernel_size=3, stride=1).cuda()
    input = torch.rand(3, 64, 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
