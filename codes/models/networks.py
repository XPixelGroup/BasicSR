import logging
import torch
import torch.nn as nn

import models.modules.SRResNet_arch as SRResNet_arch
logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            nf=opt_net['nf'],
            nb=opt_net['nb'],
            upscale=opt_net['scale'])
    # elif which_model == 'sft_arch':  # SFT-GAN
    #     netG = sft_arch.SFT_Net()
    # elif which_model == 'RRDB_net':  # RRDB
    #     netG = arch.RRDBNet(
    #         in_nc=opt_net['in_nc'],
    #         out_nc=opt_net['out_nc'],
    #         nf=opt_net['nf'],
    #         nb=opt_net['nb'],
    #         gc=opt_net['gc'],
    #         upscale=opt_net['scale'],
    #         norm_type=opt_net['norm_type'],
    #         act_type='leakyrelu',
    #         mode=opt_net['mode'],
    #         upsample_mode='upconv')
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


'''
#### Discriminator
def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])

    elif which_model == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        netD = sft_arch.ACD_VGG_BN_96()

    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, device=device)
    # netF = arch.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
'''
