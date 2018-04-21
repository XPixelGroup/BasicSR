import collections
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable


# helper selecting activation
def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


# helper selecting normalization layer
def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


# helper selecting padding layer
# if padding is 'zero', do by conv layers
def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % pad_type)
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


# Concat the output of a submodule to its input
class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


#Elementwise sum the output of a submodule to its input
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


# Flatten Sequential. It unwraps nn.Sequential.
def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0] # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


"""
Conv layer with padding, normalization, activation
mode: CNA --> Conv -> Norm -> Act
      NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
"""
def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    assert mode in ['CNA', 'NAC'], 'Wong conv mode [%s]' % mode
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if mode == 'CNA':
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


# TODO: add deconv_block here.


####################
# Useful blocks
####################
"""
ResNet Block, 3-3 style
with extra residual scaling used in EDSR
(Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
"""
class ResNetBlock(nn.Module):
    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x

        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


"""
ResNeXt Block, 1-3-1 style
(Aggregated Residual Transformations for Deep Neural Networks, CVPR17)
"""
class ResNeXtBlock(nn.Module):
    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNeXtBlock, self).__init__()
        # 1x1 conv
        conv0 = conv_block(in_nc, mid_nc, 1, stride, dilation, 1, bias, pad_type, \
                norm_type, act_type, mode)
        # 3x3 conv
        conv1 = conv_block(mid_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type,\
                norm_type, act_type, mode)
        # 1x1 conv
        if mode == 'CNA':
            act_type = None
        conv2 = conv_block(mid_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
                norm_type, act_type, mode)
        self.residual = sequential(conv0, conv1, conv2)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.residual(x).mul(self.res_scale)
        return x + res


"""
Dense Block, 5 convs style
Similar to (Image Super-Resolution Using Dense Skip Connections, ICCV 17)
"""
class DenseBlock_5C(nn.Module):
    def __init__(self, in_nc, out_nc, gc=32, kernel_size=3, stride=1, dilation=1, groups=1, \
                bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
        super(DenseBlock_5C, self).__init__()
        # gc: growth channel
        self.conv0 = conv_block(in_nc, gc, kernel_size, stride, dilation, groups, bias, pad_type,\
                norm_type, act_type, mode)
        self.conv1 = conv_block(in_nc+gc, gc, kernel_size, stride, dilation, groups, bias, \
                pad_type, norm_type, act_type, mode)
        self.conv2 = conv_block(in_nc+2*gc, gc, kernel_size, stride, dilation, groups, bias, \
                pad_type, norm_type, act_type, mode)
        self.conv3 = conv_block(in_nc+3*gc, gc, kernel_size, stride, dilation, groups, bias, \
                pad_type, norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        # merging
        self.conv4 = conv2d_block(in_nc+4*gc, out_nc, kernel_size, stride, dilation, groups, bias, \
                pad_type, norm_type, act_type, mode)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(torch.cat((x, x1), 1))
        x2 = self.conv2(torch.cat((x, x1, x2), 1))
        x3 = self.conv3(torch.cat((x, x1, x2, x3), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3, x4), 1))
        return x4


"""
Residual Dense Block, 5 convs style
The core module in (Residual Dense Network for Image Super-Resolution, CVPR 18)
"""
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, in_nc, out_nc, gc=32, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=0.1):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel
        self.conv0 = conv_block(in_nc, gc, kernel_size, stride, dilation, groups, bias, pad_type,\
                norm_type, act_type, mode)
        self.conv1 = conv_block(in_nc+gc, gc, kernel_size, stride, dilation, groups, bias, \
                pad_type, norm_type, act_type, mode)
        self.conv2 = conv_block(in_nc+2*gc, gc, kernel_size, stride, dilation, groups, bias, \
                pad_type, norm_type, act_type, mode)
        self.conv3 = conv_block(in_nc+3*gc, gc, kernel_size, stride, dilation, groups, bias, \
                pad_type, norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        # merging
        self.conv4 = conv_block(in_nc+4*gc, out_nc, kernel_size, stride, dilation, groups, bias, \
                pad_type, norm_type, act_type, mode)
        self.res_scale = res_scale

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(torch.cat((x, x1), 1))
        x2 = self.conv2(torch.cat((x, x1, x2), 1))
        x3 = self.conv3(torch.cat((x, x1, x2, x3), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3, x4), 1))
        return x4.mul(self.res_scale) + x


"""
Dual Path Basic Block, 1-3-1 style
Described in the paper: (Dual Path Networks, CVPR 17)
Code modified from github repo: rwightman/pytorch-dpn-pretrained
"""
# TODO: need to modify
# ReLU(inplace) w/o BN before Conv will cause probelm:
# one of the variables needed for gradient computation has been modified by an inplace operation
class DualPathBasicBlock_131(nn.Module):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal',
                 bias=True, pad_type=None, norm_type='batch', act_type='relu'):
        super(DualPathBasicBlock_131, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.block_type = block_type

        if self.block_type is 'proj':
            self.c1x1_w_s1 = Conv2dBlock(in_chs, num_1x1_c + inc, kernel_size=1, stride=1, bias=bias,
                    pad_type=pad_type, norm_type=norm_type, act_type=act_type, groups=1)
        self.c1x1_a = Conv2dBlock(in_chs, num_1x1_a, kernel_size=1, stride=1, bias=bias,
                    pad_type=pad_type, norm_type=norm_type, act_type=act_type, groups=1)
        self.c3x3_b = Conv2dBlock(num_1x1_a, num_3x3_b, kernel_size=3, stride=1, bias=bias, groups=groups,
                    pad_type=pad_type, norm_type=norm_type, act_type=act_type)
        self.c1x1_c = Conv2dBlock(num_3x3_b, num_1x1_c + inc, kernel_size=1, stride=1, bias=bias,
                    pad_type=pad_type, norm_type=norm_type, act_type=act_type, groups=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.block_type is 'proj':
            x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        x_in = self.c1x1_c(x_in)
        out1 = x_in[:, :self.num_1x1_c, :, :]
        out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


"""
Dual Path Block
Based on Dual Path Base Block
"""
# TODO: need to modify
class DualPathBlock(nn.Module):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, n_basic_block,
                 bias=True, pad_type=None, norm_type='batch', act_type='relu'):
        super(DualPathBlock, self).__init__()
        in_chs_his = in_chs
        assert n_basic_block > 1, 'n_basic_block in DualPathBlock must larger than 1.'
        blocks = OrderedDict()
        blocks['DPBB1'] = DualPathBasicBlock_131(in_chs=in_chs, num_1x1_a=num_1x1_a, num_3x3_b=num_3x3_b, num_1x1_c=num_1x1_c,
                inc=inc, groups=groups, block_type='proj', bias=bias, pad_type=pad_type, norm_type =norm_type, act_type=act_type)
        # the first will increase two inc
        in_chs += 2*inc
        for i in range(2, n_basic_block+1):
            blocks['DPBB' + str(i)] = DualPathBasicBlock_131(in_chs=in_chs, num_1x1_a=num_1x1_a, num_3x3_b=num_3x3_b, num_1x1_c=num_1x1_c,
                    inc=inc, groups=groups, block_type='normal', bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type)
            in_chs += inc
        self.compression_conv = Conv2dBlock(in_chs, in_chs_his, kernel_size=1, stride=1, dilation=1, bias=bias,
                 pad_type=pad_type, norm_type=norm_type, act_type=None)
        self.DPBB_blocks = nn.Sequential(blocks)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        x_in = self.DPBB_blocks(x_in)
        x_in2 = torch.cat(x_in, dim=1) if isinstance(x_in, tuple) else x_in
        return self.compression_conv(x_in2)


"""
Residual Dual Path Block
Based on Dual Path Base Block
"""
# TODO: need to modify
class ResidualDualPathBlock(nn.Module):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, n_basic_block,
                 bias=True, pad_type=None, norm_type='batch', act_type='relu'):
        super(ResidualDualPathBlock, self).__init__()
        in_chs_his = in_chs
        assert n_basic_block > 1, 'n_basic_block in DualPathBlock must larger than 1.'
        blocks = OrderedDict()
        blocks['DPBB1'] = DualPathBasicBlock_131(in_chs=in_chs, num_1x1_a=num_1x1_a, num_3x3_b=num_3x3_b, num_1x1_c=num_1x1_c,
                inc=inc, groups=groups, block_type='proj', bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type)
        # the first will increase two inc
        in_chs += 2*inc
        for i in range(2, n_basic_block+1):
            blocks['DPBB' + str(i)] = DualPathBasicBlock_131(in_chs=in_chs, num_1x1_a=num_1x1_a, num_3x3_b=num_3x3_b, num_1x1_c=num_1x1_c,
                    inc=inc, groups=groups, block_type='normal', bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type)
            in_chs += inc
        self.compression_conv = Conv2dBlock(in_chs, in_chs_his, kernel_size=1, stride=1, dilation=1, bias=bias,
                 pad_type=pad_type, norm_type=norm_type, act_type=None)
        self.DPBB_blocks = nn.Sequential(blocks)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        x_in = self.DPBB_blocks(x_in)
        x_in2 = torch.cat(x_in, dim=1) if isinstance(x_in, tuple) else x_in
        x_2 = self.compression_conv(x_in2)
        return x_2.mul(0.1) + x


####################
# Upsampler
####################
# Pixel shuffle layer
# (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional \
#   Neural Network, CVPR17)
def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                        pad_type='zero', norm_type=None, act_type='relu'):
    conv = conv_block(in_nc, out_nc*(upscale_factor**2), kernel_size, stride, bias=bias,
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


# Up conv
# describe in https://distill.pub/2016/deconv-checkerboard/
def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                        pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias,
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)


####################
# backups
####################

# MeanShift
# Used in EDSR, substract the mean of DIV2K training images
# TODO: need to modify
class MeanShift(nn.Conv2d):  # inherited from Conv2d
    def __init__(self, rgb_range, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) * rgb_range
        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False
