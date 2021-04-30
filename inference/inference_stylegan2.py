import argparse
import math
import os
import torch
from torchvision import utils

from basicsr.archs.stylegan2_arch import StyleGAN2Generator
from basicsr.utils import set_random_seed


def generate(args, g_ema, device, mean_latent, randomize_noise):

    with torch.no_grad():
        g_ema.eval()
        for i in range(args.pics):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema([sample_z],
                              truncation=args.truncation,
                              randomize_noise=randomize_noise,
                              truncation_latent=mean_latent)

            utils.save_image(
                sample,
                f'samples/{str(i).zfill(6)}.png',
                nrow=int(math.sqrt(args.sample)),
                normalize=True,
                range=(-1, 1),
            )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=1)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument(
        '--ckpt',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/StyleGAN/stylegan2_ffhq_config_f_1024_official-3ab41b38.pth'  # noqa: E501
    )
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--randomize_noise', type=bool, default=True)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
    os.makedirs('samples', exist_ok=True)
    set_random_seed(2020)

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

    generate(args, g_ema, device, mean_latent, args.randomize_noise)
