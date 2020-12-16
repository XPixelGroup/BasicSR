import torch
import torch.nn as nn


class MeanShift(nn.Conv2d):
    """ Data normalization with mean and std.

    Args:
        rgb_range (int): Maximum value of RGB.
        rgb_mean (list): Mean for RGB channels.
        rgb_std (list): Std for RGB channels.
        sign (int): for substraction, sign is -1, for addition, sign is 1.
    """

    def __init__(self,
                 rgb_range,
                 rgb_mean,
                 rgb_std,
                 sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class ResidualBlock(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-ReLU-
         |________________|
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.body(x)
        out = self.relu(out + x)
        return out


class EResidualBlock(nn.Module):
    """Enhanced Residual block.

    There are three convolution layers in residual branch.

    It has a style of:
        ---Conv-ReLU-Conv-ReLU-Conv-+-ReLU-
         |__________________________|
    """

    def __init__(self, in_channels, out_channels):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.body(x)
        out = self.relu(out + x)
        return out


class MergeRun(nn.Module):
    """ Merge-and-run unit.

    This unit contains two branches with different dilated convolutions,
    followed by a convolution to process the concatenated features.

    Paper: Real Image Denoising with Feature Attention
    Ref git repo: https://github.com/saeed-anwar/RIDNet
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(MergeRun, self).__init__()

        self.dilation1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 3, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, 4, 4),
            nn.ReLU(inplace=True)
        )

        self.aggregation = nn.Sequential(
            nn.Conv2d(
                out_channels * 2, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        dilation1 = self.dilation1(x)
        dilation2 = self.dilation2(x)
        out = torch.cat([dilation1, dilation2], dim=1)
        out = self.aggregation(out)
        out = out + x
        return out
