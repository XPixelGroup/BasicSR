"""
architecture for sft, torch version
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SFTLayer_torch(nn.Module):
    def __init__(self):
        super(SFTLayer_torch, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.01, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.01, inplace=True))
        return x[0] * scale + shift


class ResBlock_SFT_torch(nn.Module):
    def __init__(self):
        super(ResBlock_SFT_torch, self).__init__()
        self.sft0 = SFTLayer_torch()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer_torch()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = F.relu(self.sft0(x), inplace=True)
        fea = self.conv0(fea)
        fea = F.relu(self.sft1((fea, x[1])), inplace=True)
        fea = self.conv1(fea)
        return (x[0] + fea, x[1]) # return a tuple containing features and conditions


class SFT_Net_torch(nn.Module):
    def __init__(self):
        super(SFT_Net_torch, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)

        sft_branch = []
        for i in range(16):
            sft_branch.append(ResBlock_SFT_torch())
        sft_branch.append(SFTLayer_torch())
        sft_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)

        self.HR_branch = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

        # Condtion network
        self.CondNet = nn.Sequential(
            nn.Conv2d(8, 128, 4, 4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 1)
        )

    def forward(self, x):
        # x[0]: img; x[1]: seg
        cond = self.CondNet(x[1])
        fea = self.conv0(x[0])
        res = self.sft_branch((fea, cond))
        fea = fea + res
        out = self.HR_branch(fea)
        return out
