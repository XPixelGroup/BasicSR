import torch

from basicsr.archs.stylegan2_arch import (StyleGAN2Discriminator,
                                          StyleGAN2Generator)


def convert_net_g(ori_net, crt_net):
    """Convert network generator."""

    for crt_k, crt_v in crt_net.items():
        if 'style_mlp' in crt_k:
            ori_k = crt_k.replace('style_mlp', 'style')
        elif 'constant_input.weight' in crt_k:
            ori_k = crt_k.replace('constant_input.weight', 'input.input')
        # style conv1
        elif 'style_conv1.modulated_conv' in crt_k:
            ori_k = crt_k.replace('style_conv1.modulated_conv', 'conv1.conv')
        elif 'style_conv1' in crt_k:
            if crt_v.shape == torch.Size([1]):
                ori_k = crt_k.replace('style_conv1', 'conv1.noise')
            else:
                ori_k = crt_k.replace('style_conv1', 'conv1')
        # style conv
        elif 'style_convs' in crt_k:
            ori_k = crt_k.replace('style_convs',
                                  'convs').replace('modulated_conv', 'conv')
            if crt_v.shape == torch.Size([1]):
                ori_k = ori_k.replace('.weight', '.noise.weight')
        # to_rgb1
        elif 'to_rgb1.modulated_conv' in crt_k:
            ori_k = crt_k.replace('to_rgb1.modulated_conv', 'to_rgb1.conv')
        # to_rgbs
        elif 'to_rgbs' in crt_k:
            ori_k = crt_k.replace('modulated_conv', 'conv')
        elif 'noises' in crt_k:
            ori_k = crt_k.replace('.noise', '.noise_')
        else:
            ori_k = crt_k

        # replace
        if crt_net[crt_k].size() != ori_net[ori_k].size():
            raise ValueError('Wrong tensor size: \n'
                             f'crt_net: {crt_net[crt_k].size()}\n'
                             f'ori_net: {ori_net[ori_k].size()}')
        else:
            crt_net[crt_k] = ori_net[ori_k]

    return crt_net


def convert_net_d(ori_net, crt_net):
    """Convert network discriminator."""

    for crt_k, crt_v in crt_net.items():
        if 'conv_body' in crt_k:
            ori_k = crt_k.replace('conv_body', 'convs')
        else:
            ori_k = crt_k

        # replace
        if crt_net[crt_k].size() != ori_net[ori_k].size():
            raise ValueError('Wrong tensor size: \n'
                             f'crt_net: {crt_net[crt_k].size()}\n'
                             f'ori_net: {ori_net[ori_k].size()}')
        else:
            crt_net[crt_k] = ori_net[ori_k]
    return crt_net


if __name__ == '__main__':
    """Convert official stylegan2 weights from stylegan2-pytorch."""

    # configuration
    ori_net = torch.load('experiments/pretrained_models/stylegan2-ffhq.pth')
    save_path_g = 'experiments/pretrained_models/stylegan2_ffhq_config_f_1024_official.pth'  # noqa: E501
    save_path_d = 'experiments/pretrained_models/stylegan2_ffhq_config_f_1024_discriminator_official.pth'  # noqa: E501
    out_size = 1024
    channel_multiplier = 1

    # convert generator
    crt_net = StyleGAN2Generator(
        out_size,
        num_style_feat=512,
        num_mlp=8,
        channel_multiplier=channel_multiplier)
    crt_net = crt_net.state_dict()

    crt_net_params_ema = convert_net_g(ori_net['g_ema'], crt_net)
    torch.save(
        dict(params_ema=crt_net_params_ema, latent_avg=ori_net['latent_avg']),
        save_path_g)

    # convert discriminator
    crt_net = StyleGAN2Discriminator(
        out_size, channel_multiplier=channel_multiplier)
    crt_net = crt_net.state_dict()

    crt_net_params = convert_net_d(ori_net['d'], crt_net)
    torch.save(dict(params=crt_net_params), save_path_d)
