import torch
import torchvision
from distutils.version import LooseVersion
from importlib import import_module
from torch.nn import functional as F

dcn = import_module('basicsr.ops.dcn.deform_conv')


def test_dcn_torchversion():
    assert LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0')
    assert dcn.deform_conv_ext is not None
    assert dcn.deform_conv != dcn.deform_conv_tvops
    assert dcn.modulated_deform_conv != dcn.modulated_deform_conv_tvops


def test_DeformConv():

    class DeformConv_tvops(dcn.DeformConv):

        def forward(self, x, offset):
            input_pad = (x.size(2) < self.kernel_size[0] or x.size(3) < self.kernel_size[1])
            if input_pad:
                pad_h = max(self.kernel_size[0] - x.size(2), 0)
                pad_w = max(self.kernel_size[1] - x.size(3), 0)
                x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
                offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            out = dcn.deform_conv_tvops(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups,
                                        self.deformable_groups)
            if input_pad:
                out = out[:, :, :out.size(2) - pad_h, :out.size(3) - pad_w].contiguous()
            return out

    batch_size = 4
    patch_size = 16
    in_ch = 8
    out_ch = 8
    kernel_size = 3
    stride = 1
    padding = 1
    deform_group = 2

    a = dcn.DeformConv(in_ch, out_ch, kernel_size, stride, padding, deformable_groups=deform_group).cuda()
    b = DeformConv_tvops(in_ch, out_ch, kernel_size, stride, padding, deformable_groups=deform_group).cuda()

    for a_param, b_param in zip(a.parameters(), b.parameters()):
        b_param.data = a_param.data.detach().clone()

    x = torch.randn(batch_size, in_ch, patch_size, patch_size).cuda()
    offset = torch.randn(batch_size, deform_group * 2 * kernel_size * kernel_size, patch_size, patch_size).cuda()

    y1 = a(x, offset)
    y2 = b(x, offset)

    assert True == ((y1 == y2).all())


def test_ModulatedDeformConv():

    class ModulatedDeformConv_tvops(dcn.ModulatedDeformConv):

        def forward(self, x, offset, mask):
            return dcn.modulated_deform_conv_tvops(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                                   self.dilation, self.groups, self.deformable_groups)

    batch_size = 4
    patch_size = 16
    in_ch = 8
    out_ch = 8
    kernel_size = 3
    stride = 1
    padding = 1
    deform_group = 2

    a = dcn.ModulatedDeformConv(in_ch, out_ch, kernel_size, stride, padding, deformable_groups=deform_group).cuda()
    b = ModulatedDeformConv_tvops(in_ch, out_ch, kernel_size, stride, padding, deformable_groups=deform_group).cuda()

    for a_param, b_param in zip(a.parameters(), b.parameters()):
        b_param.data = a_param.data.detach().clone()

    x = torch.randn(batch_size, in_ch, patch_size, patch_size).cuda()
    offset = torch.randn(batch_size, deform_group * 2 * kernel_size * kernel_size, patch_size, patch_size).cuda()
    mask = torch.randn(batch_size, deform_group * kernel_size * kernel_size, patch_size, patch_size).cuda()

    y1 = a(x, offset, mask)
    y2 = b(x, offset, mask)

    assert True == ((y1 == y2).all())
