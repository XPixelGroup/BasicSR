import argparse
import mmcv
import torch
from torchvision import utils

from basicsr.models.archs.stylegan2_arch import StyleGAN2Generator


def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in range(args.pics):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema([sample_z],
                              truncation=args.truncation,
                              randomize_noise=True,
                              truncation_latent=mean_latent)

            utils.save_image(
                sample,
                f'samples/{str(i).zfill(6)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument(
        '--ckpt',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/stylegan2_ffhq_config_f_1024_official-f8a4b805.pth'  # noqa: E501
    )
    parser.add_argument('--channel_multiplier', type=int, default=2)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
    mmcv.mkdir_or_exist('samples')

    g_ema = StyleGAN2Generator(
        args.size,
        args.latent,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier).to(device)
    checkpoint = torch.load(args.ckpt)['params_ema']

    g_ema.load_state_dict(checkpoint)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
