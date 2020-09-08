import torch

from basicsr.models.archs.dfdnet_arch import DFDNet
from basicsr.models.archs.vgg_arch import NAMES


def convert_net(ori_net, crt_net):

    for crt_k, crt_v in crt_net.items():
        # vgg feature extractor
        if 'vgg_extractor' in crt_k:
            ori_k = crt_k.replace('vgg_extractor',
                                  'VggExtract').replace('vgg_net', 'model')
            if 'mean' in crt_k:
                ori_k = ori_k.replace('mean', 'RGB_mean')
            elif 'std' in crt_k:
                ori_k = ori_k.replace('std', 'RGB_std')
            else:
                idx = NAMES['vgg19'].index(crt_k.split('.')[2])
                if 'weight' in crt_k:
                    ori_k = f'VggExtract.model.features.{idx}.weight'
                else:
                    ori_k = f'VggExtract.model.features.{idx}.bias'
        elif 'attn_blocks' in crt_k:
            if 'left_eye' in crt_k:
                ori_k = crt_k.replace('attn_blocks.left_eye', 'le')
            elif 'right_eye' in crt_k:
                ori_k = crt_k.replace('attn_blocks.right_eye', 're')
            elif 'mouth' in crt_k:
                ori_k = crt_k.replace('attn_blocks.mouth', 'mo')
            elif 'nose' in crt_k:
                ori_k = crt_k.replace('attn_blocks.nose', 'no')
            else:
                raise ValueError('Wrong!')
        elif 'multi_scale_dilation' in crt_k:
            if 'conv_blocks' in crt_k:
                a, b, c, d, e = crt_k.split('.')
                ori_k = f'MSDilate.conv{int(c)+1}.{d}.{e}'
            else:
                ori_k = crt_k.replace('multi_scale_dilation.conv_fusion',
                                      'MSDilate.convi')

        elif crt_k.startswith('upsample'):
            ori_k = crt_k.replace('upsample', 'up')
            if 'scale_block' in crt_k:
                ori_k = ori_k.replace('scale_block', 'ScaleModel1')
            elif 'shift_block' in crt_k:
                ori_k = ori_k.replace('shift_block', 'ShiftModel1')

            elif 'upsample4' in crt_k and 'body' in crt_k:
                ori_k = ori_k.replace('body', 'Model')

        else:
            print('unprocess key: ', crt_k)

        # replace
        if crt_net[crt_k].size() != ori_net[ori_k].size():
            raise ValueError('Wrong tensor size: \n'
                             f'crt_net: {crt_net[crt_k].size()}\n'
                             f'ori_net: {ori_net[ori_k].size()}')
        else:
            crt_net[crt_k] = ori_net[ori_k]

    return crt_net


if __name__ == '__main__':
    ori_net = torch.load(
        'experiments/pretrained_models/DFDNet/DFDNet_official_original.pth')
    dfd_net = DFDNet(
        64,
        dict_path='experiments/pretrained_models/DFDNet/DFDNet_dict_512.pth')
    crt_net = dfd_net.state_dict()
    crt_net_params = convert_net(ori_net, crt_net)

    torch.save(
        dict(params=crt_net_params),
        'experiments/pretrained_models/DFDNet/DFDNet_official.pth',
        _use_new_zipfile_serialization=False)
