"""
architecture for sft
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from . import block as B
import block as B # use for unit test


class ConditionNet(nn.Module):
    def __init__(self):
        super(ConditionNet, self).__init__()
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
        x = self.CondNet(x)
        return x


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.01, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.01, inplace=True))
        return x[0] * scale + shift


class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = F.relu(self.sft0(x), inplace=True)
        fea = self.conv0(fea)
        fea = F.relu(self.sft1((fea, x[1])), inplace=True)
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

    def forward(self, x):
        # x[0]: img; x[1]: cond
        fea = self.conv0(x[0])
        res = self.sft_branch((fea, x[1]))
        fea = fea + res
        out = self.HR_branch(fea)
        return out

if __name__ == '__main__':
    import os.path
    import cv2
    import numpy as np
    from torch.autograd import Variable
    import sys
    sys.path.insert(0, os.path.abspath('../../'))
    from data.util import imresize
    import torchvision.utils

    # condition network
    cond_net = ConditionNet()
    # # save model
    # save_path = '/home/xtwang/Projects/BasicSR/torch_to_pytorch/sft/cond_raw.pth'
    # state_dict = cond_net.state_dict()
    # torch.save(state_dict, save_path)
    load_path = '../../../experiments/pretrained_models/condition_net.pth'
    cond_net.load_state_dict(torch.load(load_path), strict=True)
    cond_net.eval()

    # sft network
    sft_net = SFT_Net()
    # # save model
    # save_path = '/home/xtwang/Projects/BasicSR/torch_to_pytorch/sft/sft_raw.pth'
    # state_dict = sft_net.state_dict()
    # torch.save(state_dict, save_path)
    load_path = '../../../experiments/pretrained_models/sft_net.pth'
    sft_net.load_state_dict(torch.load(load_path), strict=True)
    sft_net.eval()

    print('testing...')
    cond_net = cond_net.cuda()
    sft_net = sft_net.cuda()
    seg = torch.load('../../../data/samples_segprob/OST_013_bic.pth')
    seg = seg.cuda()
    shared_cond = cond_net(Variable(seg, volatile=True))

    img = cv2.imread('../../../data/samples/OST_013.png', cv2.IMREAD_UNCHANGED)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # matlab imresize
    img_LR = imresize(img, 1 / 4, antialiasing=True)
    img_LR = img_LR.unsqueeze(0)
    img_LR = img_LR.cuda()

    output = sft_net((Variable(img_LR, volatile=True), shared_cond)).data.float().cpu()
    output.squeeze_()
    torchvision.utils.save_image(output, 'rlt.png', padding=0, normalize=False)
