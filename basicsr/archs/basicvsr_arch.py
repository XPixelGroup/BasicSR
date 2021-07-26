import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .spynet_arch import SpyNet


@ARCH_REGISTRY.register()
class BasicVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=30, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_back = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_for = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_for, flows_back

    def forward(self, x):
        flows_for, flows_back = self.get_flow(x)

        b, n, c, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_back[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_for[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)
