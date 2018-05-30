"""
architecture for segmentation
"""
import torch
import torch.nn as nn
from . import block as B
# import block as B # use for unit test


class Res131(nn.Module):
    def __init__(self, in_nc, mid_nc, out_nc, dilation=1, stride=1):
        super(Res131, self).__init__()
        conv0 = B.conv_block(in_nc, mid_nc, 1, 1, 1, 1, False, 'zero', 'batch')
        conv1 = B.conv_block(mid_nc, mid_nc, 3, stride, dilation, 1, False, 'zero', 'batch')
        conv2 = B.conv_block(mid_nc, out_nc, 1, 1, 1, 1, False, 'zero', 'batch', None) #  No ReLU
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
        conv1_1 = B.conv_block(3, 64, 3, 2, 1, 1, False, 'zero', 'batch') #  /2
        conv1_2 = B.conv_block(64, 64, 3, 1, 1, 1, False, 'zero', 'batch')
        conv1_3 = B.conv_block(64, 128, 3, 1, 1, 1, False, 'zero', 'batch')
        max_pool = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True) #  /2
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


if __name__ == '__main__':
    import os.path
    import cv2
    import numpy as np
    from torch.autograd import Variable

    # save model(for further transfer weights from t7 to pth)
    # save_path = '/home/xtwang/Projects/BasicSR/torch_to_pytorch/pytorch_models/OutdoorSceneSeg_bic_iter_30000.pth'
    # state_dict = net.state_dict()
    # torch.save(state_dict, save_path)

    # load network
    net = OutdoorSceneSeg()
    load_path = '../../../experiments/pretrained_models/OutdoorSceneSeg_bic.pth'
    net.load_state_dict(torch.load(load_path), strict=True)
    net.eval()

    # test
    net = net.cuda()
    # read image
    img = cv2.imread('../../../data/examples/OST_013.png', cv2.IMREAD_UNCHANGED)
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()  # BGR, [0, 255]
    img[0] -= 103.939
    img[1] -= 116.779
    img[2] -= 123.68
    img = img.unsqueeze(0)
    img = img.cuda()
    img_input = Variable(img, volatile=True)
    output = net(img_input)
    print(output)
