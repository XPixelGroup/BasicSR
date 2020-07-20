import copy
import random
import torch
from torch import nn as nn


class ToyDiscriminator(nn.Module):

    def __init__(self):
        super(ToyDiscriminator, self).__init__()
        self.conv0 = nn.Conv2d(3, 4, 3, 1, 1, bias=True)
        self.bn0 = nn.BatchNorm2d(4, affine=True)
        self.conv1 = nn.Conv2d(4, 4, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(4, affine=True)
        self.linear = nn.Linear(4 * 6 * 6, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.bn0(self.conv0(x)))
        feat = self.lrelu(self.bn1(self.conv1(feat)))
        feat = feat.view(feat.size(0), -1)
        out = torch.sigmoid(self.linear(feat))
        return out


def main():
    # use fixed random seed
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    img_real = torch.rand((1, 3, 6, 6))
    img_fake = torch.rand((1, 3, 6, 6))
    net_d_1 = ToyDiscriminator()
    net_d_2 = copy.deepcopy(net_d_1)
    net_d_1.train()
    net_d_2.train()

    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0

    for k, v in net_d_1.named_parameters():
        print(k, v.size())

    ###########################
    # (1) Backward D network twice as the official tutorial does:
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    ###########################
    net_d_1.zero_grad()
    # real
    output = net_d_1(img_real).view(-1)
    label = output.new_ones(output.size()) * real_label
    loss_real = criterion(output, label)
    loss_real.backward()
    # fake
    output = net_d_1(img_fake).view(-1)
    label = output.new_ones(output.size()) * fake_label
    loss_fake = criterion(output, label)
    loss_fake.backward()

    ###########################
    # (2) Backward D network once
    ###########################
    net_d_2.zero_grad()
    # real
    output = net_d_2(img_real).view(-1)
    label = output.new_ones(output.size()) * real_label
    loss_real = criterion(output, label)
    # fake
    output = net_d_2(img_fake).view(-1)
    label = output.new_ones(output.size()) * fake_label
    loss_fake = criterion(output, label)

    loss = loss_real + loss_fake
    loss.backward()

    ###########################
    # Compare differences
    ###########################
    for k1, k2 in zip(net_d_1.parameters(), net_d_2.parameters()):
        print(torch.sum(torch.abs(k1.grad - k2.grad)))


if __name__ == '__main__':
    main()
r"""Output:
conv0.weight torch.Size([4, 3, 3, 3])
conv0.bias torch.Size([4])
bn0.weight torch.Size([4])
bn0.bias torch.Size([4])
conv1.weight torch.Size([4, 4, 3, 3])
conv1.bias torch.Size([4])
bn1.weight torch.Size([4])
bn1.bias torch.Size([4])
linear.weight torch.Size([1, 144])
linear.bias torch.Size([1])
tensor(0.)
tensor(0.)
tensor(0.)
tensor(0.)
tensor(0.)
tensor(0.)
tensor(0.)
tensor(0.)
tensor(0.)
tensor(0.)
"""
