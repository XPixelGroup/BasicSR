import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import os
import re
import copy  # deepcopy
from basicsr.models.archs.sync_batchnorm import SynchronizedBatchNorm2d

# [Lotayou 20210424] Warning: spectral norm could be buggy
# under eval mode and multi-GPU inference
# A workaround is sticking to single-GPU inference and train mode
import torch.nn.utils.spectral_norm as spectral_norm
'''
    Code migrated from SPADE project
'''
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class InstanceNorm2d(nn.Module):
    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def forward(self, x):
        #x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = InstanceNorm2d(norm_nc)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128 if norm_nc>128 else norm_nc

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, bias=False)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, bias=False)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * gamma + beta

        return out


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, semantic_nc=None):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        if semantic_nc is None:
            semantic_nc = opt.semantic_nc

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out



'''
    Arch definition begin
'''
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
                

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh, self.scale_ratio = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.ups = nn.ModuleList([
            SPADEResnetBlock(16 * nf, 8 * nf, opt),
            SPADEResnetBlock(8 * nf, 4 * nf, opt),
            SPADEResnetBlock(4 * nf, 2 * nf, opt),
            SPADEResnetBlock(2 * nf, 1 * nf, opt)  # here
            ])

        self.to_rgbs = nn.ModuleList([
            nn.Conv2d(8 * nf, 3, 3, padding=1),
            nn.Conv2d(4 * nf, 3, 3, padding=1),
            nn.Conv2d(2 * nf, 3, 3, padding=1),
            nn.Conv2d(1 * nf, 3, 3, padding=1)      # here
            ])

        self.up = nn.Upsample(scale_factor=2)
        
    # 20200309 interface for flexible encoder design
    # and mid-level loss control!
    # For basic network, it's just a 16x downsampling
    def encode(self, input):
        h, w = input.size()[-2:]
        sh, sw = h//2**self.scale_ratio, w//2**self.scale_ratio
        x = F.interpolate(input, size=(sh, sw))
        return self.fc(x) # 20200310: Merge fc into encoder

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        # 20200211 Yang Lingbo with respect to phase
        scale_ratio = num_up_layers
        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh, scale_ratio

    def forward(self, input, seg=None):
        if seg is None:
            seg = input # Interesting change...
            
        # For basic generator, 16x downsampling.
        # 20200310: Merge fc into encoder
        x = self.encode(input)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        
        if self.opt.is_test:
            phase = len(self.to_rgbs)
        else:
            phase = self.opt.train_phase+1

        for i in range(phase):
            x = self.up(x)
            x = self.ups[i](x, seg)
        
        x = self.to_rgbs[phase-1](F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x
        
    def mixed_guidance_forward(self, input, seg=None, n=0, mode='progressive'):
        '''
            A helper class for subspace visualization.
            Input and seg are different images
            For the first n levels (including encoder)
            we use input, for the rest we use seg.
            
            If mode = 'progressive', the output's like: AAABBB
            If mode = 'one_plug', the output's like:    AAABAA
            If mode = 'one_ablate', the output's like:  BBBABB
        '''
        
        if seg is None:
            return self.forward(input)
            
        if self.opt.is_test:
            phase = len(self.to_rgbs)
        else:
            phase = self.opt.train_phase+1
        
        if mode == 'progressive':
            n = max(min(n, 4 + phase), 0)
            guide_list = [input] * n + [seg] * (4+phase-n)
        elif mode == 'one_plug':
            n = max(min(n, 4 + phase-1), 0)
            guide_list = [seg] * (4+phase)
            guide_list[n] = input
        elif mode == 'one_ablate':
            if n > 3+phase:
                return self.forward(input)
            guide_list = [input] * (4+phase)
            guide_list[n] = seg
        
        x = self.encode(guide_list[0])
        x = self.head_0(x, guide_list[1])

        x = self.up(x)
        x = self.G_middle_0(x, guide_list[2])
        x = self.G_middle_1(x, guide_list[3])
        
        for i in range(phase):
            x = self.up(x)
            x = self.ups[i](x, guide_list[4+i])
        
        x = self.to_rgbs[phase-1](F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x
        

def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x*weight, kernel, stride, padding) / F.avg_pool2d(weight, kernel, stride, padding)

class SoftGate(nn.Module):
    COEFF = 12.0
    
    def __init__(self):
        super(SoftGate, self).__init__()
    
    def forward(self, x):
        return torch.sigmoid(x).mul(self.COEFF)

class SimplifiedLIP(nn.Module):
    def __init__(self, channels):
        super(SimplifiedLIP, self).__init__()

        rp = channels

        self.logit = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            SoftGate()
        )
        
    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x))
        return frac
        
class LIPEncoder(BaseNetwork):
    def __init__(self, opt, sw, sh, n_2xdown,
            norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.sw = sw
        self.sh = sh
        self.max_ratio = 16
        
        # norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        
        # 20200310: Several Convolution (stride 1) + LIP blocks, 4 fold
        ngf = opt.ngf
        kw = 3
        pw = (kw - 1) // 2
        
        model = [
            nn.Conv2d(opt.semantic_nc, ngf, kw, stride=1, padding=pw, bias=False),
            norm_layer(ngf),
            nn.ReLU(),
        ]
        cur_ratio = 1
        for i in range(n_2xdown):
            next_ratio = min(cur_ratio*2, self.max_ratio)
            model += [
                SimplifiedLIP(ngf*cur_ratio),
                nn.Conv2d(ngf*cur_ratio, ngf*next_ratio, kw, stride=1, padding=pw),
                norm_layer(ngf*next_ratio),
            ]
            cur_ratio = next_ratio
            if i < n_2xdown - 1: 
                model += [nn.ReLU(inplace=True)]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)


class HiFaceGAN(SPADEGenerator):
    '''
        HiFaceGAN: SPADEGenerator with a learnable feature encoder
        Current encoder design: Local Importance-based Pooling (Ziteng Gao et.al.,ICCV 2019)
    '''
    def __init__(self, opt):
        super().__init__(opt)
        self.lip_encoder = LIPEncoder(opt, self.sw, self.sh, self.scale_ratio)
        
    def encode(self, x):
        return self.lip_encoder(x)

