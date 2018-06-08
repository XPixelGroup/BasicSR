import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable


class SRCNN3group_linear(nn.Module):
    def __init__(self, nf):
        super(SRCNN3group_linear, self).__init__()
        self.reflect_pad = nn.ReflectionPad2d(4)
        self.conv1 = nn.Conv2d(1, nf, 9, 1, 0)
        self.conv2 = nn.Conv2d(nf, nf, 9, 1, 0, groups=nf)
        self.conv3 = nn.Conv2d(nf, 1, 9, 1, 0)

    def forward(self, x):
        x = self.conv1(self.reflect_pad(x))
        x = self.conv2(self.reflect_pad(x))
        x = self.conv3(self.reflect_pad(x))
        return x
