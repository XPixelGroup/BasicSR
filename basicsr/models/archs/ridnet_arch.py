import torch.nn as nn

from basicsr.models.archs.arch_util import make_layer
from basicsr.models.archs.ridnet_util import (EResidualBlock, MeanShift,
                                              MergeRun, ResidualBlock)


class ChannelAttention(nn.Module):
    """Channel attention.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default:
    """

    def __init__(self, mid_channels, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                mid_channels, mid_channels // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels // squeeze_factor, mid_channels, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class EAM(nn.Module):
    """Enhancement attention modules (EAM) in RIDNet.

    This module contains a merge-and-run unit, a residual block,
    an enhanced residual block and a feature attention unit.

    Attributes:
        merge: The merge-and-run unit.
        block1: The residual block.
        block2: The enhanced residual block.
        ca: The feature/channel attention unit.
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(EAM, self).__init__()

        self.merge = MergeRun(in_channels, mid_channels)
        self.block1 = ResidualBlock(mid_channels, mid_channels)
        self.block2 = EResidualBlock(mid_channels, out_channels)
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        out = self.merge(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.ca(out)
        return out


class RIDNet(nn.Module):
    """RIDNet: Real Image Denoising with Feature Attention.

    Ref git repo: https://github.com/saeed-anwar/RIDNet

    Args:
        in_channels (int): Channel number of inputs.
        mid_channels (int): Channel number of EAM modules.
            Default: 64.
        out_channels (int): Channel number of outputs.
        num_block (int): Number of EAM. Default: 4.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 num_block=4,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0)):
        super(RIDNet, self).__init__()

        self.sub_mean = MeanShift(img_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(img_range, rgb_mean, rgb_std, 1)

        self.head = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(
            EAM,
            num_block,
            in_channels=mid_channels,
            mid_channels=mid_channels,
            out_channels=mid_channels)
        self.tail = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.sub_mean(x)
        res = self.tail(self.body(self.relu(self.head(res))))
        res = self.add_mean(res)

        out = x + res
        return out
