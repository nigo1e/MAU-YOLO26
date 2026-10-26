import torch
import torch.nn as nn
import torch.nn.functional as F


class PfAAMLayer(nn.Module):
    def __init__(self):
        super(PfAAMLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        z = torch.mean(x, dim=1, keepdim=True)
        attention = self.sigmoid(y * z)
        return x * attention
