import argparse
import math
import numpy as np
import torch
from torch import nn

from basicsr.archs.stylegan2_arch import StyleGAN2Generator
from basicsr.metrics.fid import (calculate_fid, extract_inception_features,
                                 load_patched_inception_v3)


def calculate_stylegan2_fid():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'ckpt', type=str, help='Path to the stylegan2 checkpoint.')
    parser.add_argument(
        'fid_stats', type=str, help='Path to the dataset fid statistics.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_sample', type=int, default=50000)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    args = parser.parse_args()

    # create stylegan2 model
    generator = StyleGAN2Generator(
        out_size=args.size,
        num_style_feat=512,
        num_mlp=8,
        channel_multiplier=args.channel_multiplier,
        resample_kernel=(1, 3, 3, 1))
    generator.load_state_dict(torch.load(args.ckpt)['params_ema'])
    generator = nn.DataParallel(generator).eval().to(device)

    if args.truncation < 1:
        with torch.no_grad():
            truncation_latent = generator.mean_latent(args.truncation_mean)
    else:
        truncation_latent = None

    # inception model
    inception = load_patched_inception_v3(device)

    total_batch = math.ceil(args.num_sample / args.batch_size)

    def sample_generator(total_batch):
        for i in range(total_batch):
            with torch.no_grad():
                latent = torch.randn(args.batch_size, 512, device=device)
                samples, _ = generator([latent],
                                       truncation=args.truncation,
                                       truncation_latent=truncation_latent)
            yield samples

    features = extract_inception_features(
        sample_generator(total_batch), inception, total_batch, device)
    features = features.numpy()
    total_len = features.shape[0]
    features = features[:args.num_sample]
    print(f'Extracted {total_len} features, '
          f'use the first {features.shape[0]} features to calculate stats.')
    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    # load the dataset stats
    stats = torch.load(args.fid_stats)
    real_mean = stats['mean']
    real_cov = stats['cov']

    # calculate FID metric
    fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)
    print('fid:', fid)


if __name__ == '__main__':
    calculate_stylegan2_fid()
