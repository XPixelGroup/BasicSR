"""
architecture for sft
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_norm import spectral_norm


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1]) # return a tuple containing features and conditions


class SFT_Net(nn.Module):
    def __init__(self):
        super(SFT_Net, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)

        sft_branch = []
        for i in range(16):
            sft_branch.append(ResBlock_SFT())
        sft_branch.append(SFTLayer())
        sft_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)

        self.HR_branch = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

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


# Auxiliary Classifier Discriminator
class ACD_VGG_BN_128(nn.Module):
    def __init__(self):
        super(ACD_VGG_BN_128, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(512, 512, 4, 2, 1)
        )

        # gan
        self.gan = nn.Sequential(
            nn.Linear(512*4*4, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 1)
        )

        self.cls = nn.Sequential(
            nn.Linear(512*4*4, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 7)
        )

    def forward(self, x):
        fea = self.feature(x)
        fea = fea.view(fea.size(0), -1)
        gan = self.gan(fea)
        cls = self.cls(fea)
        return [gan, cls]


class ACD_VGG_BN_96(nn.Module):
    def __init__(self):
        super(ACD_VGG_BN_96, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),
        )

        # gan
        self.gan = nn.Sequential(
            nn.Linear(512*6*6, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 1)
        )

        self.cls = nn.Sequential(
            nn.Linear(512*6*6, 7),
        )

    def forward(self, x):
        fea = self.feature(x)
        fea = fea.view(fea.size(0), -1)
        gan = self.gan(fea)
        cls = self.cls(fea)
        return [gan, cls]

class ACD_VGG_BN_128_SN(nn.Module):  # with spectral_norm
    def __init__(self):
        super(ACD_VGG_BN_128_SN, self).__init__()

        self.feature = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 3, 1, 1)),
            nn.LeakyReLU(0.1, True),

            spectral_norm(nn.Conv2d(64, 64, 4, 2, 1)),
            nn.LeakyReLU(0.1, True),

            spectral_norm(nn.Conv2d(64, 128, 3, 1, 1)),
            nn.LeakyReLU(0.1, True),

            spectral_norm(nn.Conv2d(128, 128, 4, 2, 1)),
            nn.LeakyReLU(0.1, True),

            spectral_norm(nn.Conv2d(128, 256, 3, 1, 1)),
            nn.LeakyReLU(0.1, True),

            spectral_norm(nn.Conv2d(256, 256, 4, 2, 1)),
            nn.LeakyReLU(0.1, True),

            spectral_norm(nn.Conv2d(256, 512, 3, 1, 1)),
            nn.LeakyReLU(0.1, True),

            spectral_norm(nn.Conv2d(512, 512, 4, 2, 1)),
            nn.LeakyReLU(0.1, True),

            spectral_norm(nn.Conv2d(512, 512, 3, 1, 1)),
            nn.LeakyReLU(0.1, True),

            spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        )

        # gan
        self.gan = nn.Sequential(
            spectral_norm(nn.Linear(512*4*4, 100)),
            nn.LeakyReLU(0.1, True),
            spectral_norm(nn.Linear(100, 1))
        )

        self.cls = nn.Sequential(
            spectral_norm(nn.Linear(512*4*4, 100)),
            nn.LeakyReLU(0.1, True),
            spectral_norm(nn.Linear(100, 7))
        )

    def forward(self, x):
        fea = self.feature(x)
        fea = fea.view(fea.size(0), -1)
        gan = self.gan(fea)
        cls = self.cls(fea)
        return [gan, cls]


if __name__ == '__main__':
    import os.path
    import cv2
    import numpy as np
    from torch.autograd import Variable
    import sys
    sys.path.insert(0, os.path.abspath('../../'))
    from data.util import imresize
    import torchvision.utils

    # sft network
    sft_net = SFT_Net()
    # # save model
    # save_path = '/home/xtwang/Projects/BasicSR/codes/scripts/sft_net_raw.pth'
    # state_dict = sft_net.state_dict()
    # torch.save(state_dict, save_path)

    load_path = '/home/xtwang/Projects/BasicSR/codes/scripts/sft_net.pth'
    sft_net.load_state_dict(torch.load(load_path), strict=True)
    sft_net.eval()
    sft_net.cuda()

    img = cv2.imread('/mnt/SSD/xtwang/BasicSR_datasets/OST/test/img/OST300/OST_013.png', cv2.IMREAD_UNCHANGED)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # matlab imresize
    img_LR = imresize(img, 1 / 4, antialiasing=True)
    img_LR = img_LR.unsqueeze(0)
    img_LR = img_LR.cuda()

    seg = torch.load('/mnt/SSD/xtwang/BasicSR_datasets/OST/test/bicseg/OST300/OST_013.pth')
    seg = seg.unsqueeze(0).cuda()

    output = sft_net((Variable(img_LR, volatile=True), Variable(seg))).data.float().cpu()
    output.squeeze_()
    torchvision.utils.save_image(output, 'rlt.png', padding=0, normalize=False)
