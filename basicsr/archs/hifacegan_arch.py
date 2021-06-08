'''
  HiFaceGAN_arch: Network definition of HiFaceGAN
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 20210519 [Lotayou]: Currently Pytorch official SyncBatchNorm only works under DDP training.
# from basicsr.models.archs.sync_batchnorm import SynchronizedBatchNorm2d
from basicsr.utils.registry import ARCH_REGISTRY
from .hifacegan_util import BaseNetwork, LIPEncoder, SPADEResnetBlock, get_nonspade_norm_layer


@ARCH_REGISTRY.register()
class SPADEGenerator(BaseNetwork):
    """Generator with SPADEResBlock"""

    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 use_vae=False,
                 z_dim=256,
                 num_upsampling_layers='normal',
                 crop_size=512,
                 norm_G='spectralspadesyncbatch3x3',
                 is_train=True,
                 init_train_phase=3):  # progressive training disabled
        super().__init__()
        self.nf = num_feat
        self.input_nc = num_in_ch
        self.is_train = is_train
        self.train_phase = init_train_phase

        if num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif num_upsampling_layers == 'more':
            num_up_layers = 6
        elif num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError(f'opt.num_upsampling_layers [{num_upsampling_layers}] not recognized')

        # 20200211 Yang Lingbo with respect to phase
        self.scale_ratio = num_up_layers
        self.sw = crop_size // (2**num_up_layers)
        self.sh = self.sw  # 20210519: By default use square image, aspect_ratio = 1.0

        if use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(z_dim, 16 * self.nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(num_in_ch, 16 * self.nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * self.nf, 16 * self.nf, norm_G)

        self.G_middle_0 = SPADEResnetBlock(16 * self.nf, 16 * self.nf, norm_G)
        self.G_middle_1 = SPADEResnetBlock(16 * self.nf, 16 * self.nf, norm_G)

        self.ups = nn.ModuleList([
            SPADEResnetBlock(16 * self.nf, 8 * self.nf, norm_G),
            SPADEResnetBlock(8 * self.nf, 4 * self.nf, norm_G),
            SPADEResnetBlock(4 * self.nf, 2 * self.nf, norm_G),
            SPADEResnetBlock(2 * self.nf, 1 * self.nf, norm_G)  # here
        ])

        self.to_rgbs = nn.ModuleList([
            nn.Conv2d(8 * self.nf, 3, 3, padding=1),
            nn.Conv2d(4 * self.nf, 3, 3, padding=1),
            nn.Conv2d(2 * self.nf, 3, 3, padding=1),
            nn.Conv2d(1 * self.nf, 3, 3, padding=1)  # here
        ])

        self.up = nn.Upsample(scale_factor=2)

    # 20200309 interface for flexible encoder design
    # and mid-level loss control!
    # For basic network, it's just a 16x downsampling
    def encode(self, input_tensor):
        h, w = input_tensor.size()[-2:]
        sh, sw = h // 2**self.scale_ratio, w // 2**self.scale_ratio
        x = F.interpolate(input_tensor, size=(sh, sw))
        return self.fc(x)  # 20200310: Merge fc into encoder

    def forward(self, x):
        # In oroginal SPADE, seg means a segmentation map, but here we use x instead.
        seg = x

        # For basic generator, 16x downsampling.
        # 20200310: Merge fc into encoder
        x = self.encode(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)

        if self.is_train:
            phase = self.train_phase + 1
        else:
            phase = len(self.to_rgbs)

        for i in range(phase):
            x = self.up(x)
            x = self.ups[i](x, seg)

        x = self.to_rgbs[phase - 1](F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x

    def mixed_guidance_forward(self, input_x, seg=None, n=0, mode='progressive'):
        """
            A helper class for subspace visualization.
            Input and seg are different images
            For the first n levels (including encoder)
            we use input, for the rest we use seg.

            If mode = 'progressive', the output's like: AAABBB
            If mode = 'one_plug', the output's like:    AAABAA
            If mode = 'one_ablate', the output's like:  BBBABB
        """

        if seg is None:
            return self.forward(input_x)

        if self.is_train:
            phase = self.train_phase + 1
        else:
            phase = len(self.to_rgbs)

        if mode == 'progressive':
            n = max(min(n, 4 + phase), 0)
            guide_list = [input_x] * n + [seg] * (4 + phase - n)
        elif mode == 'one_plug':
            n = max(min(n, 4 + phase - 1), 0)
            guide_list = [seg] * (4 + phase)
            guide_list[n] = input_x
        elif mode == 'one_ablate':
            if n > 3 + phase:
                return self.forward(input_x)
            guide_list = [input_x] * (4 + phase)
            guide_list[n] = seg

        x = self.encode(guide_list[0])
        x = self.head_0(x, guide_list[1])

        x = self.up(x)
        x = self.G_middle_0(x, guide_list[2])
        x = self.G_middle_1(x, guide_list[3])

        for i in range(phase):
            x = self.up(x)
            x = self.ups[i](x, guide_list[4 + i])

        x = self.to_rgbs[phase - 1](F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x


@ARCH_REGISTRY.register()
class HiFaceGAN(SPADEGenerator):
    """ HiFaceGAN: SPADEGenerator with a learnable feature encoder
        Current encoder design: LIPEncoder"""

    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 use_vae=False,
                 z_dim=256,
                 num_upsampling_layers='normal',
                 crop_size=512,
                 norm_G='spectralspadesyncbatch3x3',
                 is_train=True,
                 init_train_phase=3):

        super().__init__(num_in_ch, num_feat, use_vae, z_dim, num_upsampling_layers, crop_size, norm_G, is_train,
                         init_train_phase)
        self.lip_encoder = LIPEncoder(num_in_ch, num_feat, self.sw, self.sh, self.scale_ratio)

    def encode(self, input_tensor):
        return self.lip_encoder(input_tensor)


@ARCH_REGISTRY.register()
class HiFaceGANDiscriminator(BaseNetwork):
    """ Inspired by pix2pixHD multiscale discriminator.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        conditional_D (bool): Whether use conditional discriminator.
            Default: True.
        num_D (int): Number of Multiscale discriminators. Default: 3.
        n_layers_D (int): Number of downsample layers in each D. Default: 4.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
        norm_D (str): String to determine normalization layers in D.
            Choices: [spectral][instance/batch/syncbatch]
            Default: 'spectralinstance'.
        keep_features (bool): Keep intermediate features for matching loss, etc.
            Default: True.
    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        conditional_D=True,
        num_D=2,
        n_layers_D=4,
        num_feat=64,
        norm_D='spectralinstance',
        keep_features=True,
    ):
        super().__init__()
        self.num_D = num_D

        input_nc = num_in_ch
        if conditional_D:
            input_nc += num_out_ch

        for i in range(num_D):
            subnet_D = NLayerDiscriminator(input_nc, n_layers_D, num_feat, norm_D, keep_features)
            self.add_module(f'discriminator_{i}', subnet_D)

    def downsample(self, x):
        return F.avg_pool2d(x, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, x):
        result = []
        for _, D in self.named_children():
            out = D(x)
            result.append(out)
            x = self.downsample(x)

        return result


@ARCH_REGISTRY.register()
class NLayerDiscriminator(BaseNetwork):
    """Defines the PatchGAN discriminator with the specified arguments."""

    def __init__(self, input_nc, n_layers_D, num_feat, norm_D, keep_features):
        super().__init__()
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = num_feat
        self.keep_features = keep_features

        norm_layer = get_nonspade_norm_layer(norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]]

        for n in range(1, n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers_D - 1 else 2
            sequence += [[
                norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                nn.LeakyReLU(0.2, False)
            ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, x):
        results = [x]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        if self.keep_features:
            return results[1:]
        else:
            return results[-1]