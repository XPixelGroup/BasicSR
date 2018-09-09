'''
architecture for segmentation
'''
import torch.nn as nn
from . import block as B


class Res131(nn.Module):
    def __init__(self, in_nc, mid_nc, out_nc, dilation=1, stride=1):
        super(Res131, self).__init__()
        conv0 = B.conv_block(in_nc, mid_nc, 1, 1, 1, 1, False, 'zero', 'batch')
        conv1 = B.conv_block(mid_nc, mid_nc, 3, stride, dilation, 1, False, 'zero', 'batch')
        conv2 = B.conv_block(mid_nc, out_nc, 1, 1, 1, 1, False, 'zero', 'batch', None)  #  No ReLU
        self.res = B.sequential(conv0, conv1, conv2)
        if in_nc == out_nc:
            self.has_proj = False
        else:
            self.has_proj = True
            self.proj = B.conv_block(in_nc, out_nc, 1, stride, 1, 1, False, 'zero', 'batch', None)
            #  No ReLU

    def forward(self, x):
        res = self.res(x)
        if self.has_proj:
            x = self.proj(x)
        return nn.functional.relu(x + res, inplace=True)


class OutdoorSceneSeg(nn.Module):
    def __init__(self):
        super(OutdoorSceneSeg, self).__init__()
        # conv1
        blocks = []
        conv1_1 = B.conv_block(3, 64, 3, 2, 1, 1, False, 'zero', 'batch')  #  /2
        conv1_2 = B.conv_block(64, 64, 3, 1, 1, 1, False, 'zero', 'batch')
        conv1_3 = B.conv_block(64, 128, 3, 1, 1, 1, False, 'zero', 'batch')
        max_pool = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)  #  /2
        blocks = [conv1_1, conv1_2, conv1_3, max_pool]
        # conv2, 3 blocks
        blocks.append(Res131(128, 64, 256))
        for i in range(2):
            blocks.append(Res131(256, 64, 256))
        # conv3, 4 blocks
        blocks.append(Res131(256, 128, 512, 1, 2))  #  /2
        for i in range(3):
            blocks.append(Res131(512, 128, 512))
        # conv4, 23 blocks
        blocks.append(Res131(512, 256, 1024, 2))
        for i in range(22):
            blocks.append(Res131(1024, 256, 1024, 2))
        # conv5
        blocks.append(Res131(1024, 512, 2048, 4))
        blocks.append(Res131(2048, 512, 2048, 4))
        blocks.append(Res131(2048, 512, 2048, 4))
        blocks.append(B.conv_block(2048, 512, 3, 1, 1, 1, False, 'zero', 'batch'))
        blocks.append(nn.Dropout(0.1))
        # # conv6
        blocks.append(nn.Conv2d(512, 8, 1, 1))

        self.feature = B.sequential(*blocks)
        # deconv
        self.deconv = nn.ConvTranspose2d(8, 8, 16, 8, 4, 0, 8, False, 1)
        # softmax
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.feature(x)
        x = self.deconv(x)
        x = self.softmax(x)
        return x
