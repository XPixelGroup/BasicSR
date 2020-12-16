import argparse
import glob
import os

import cv2
import numpy as np
import torch
from basicsr.models.archs.ridnet_arch import RIDNet

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_path',
        type=str,
        default='datasets/denoise/RNI15')
    parser.add_argument('--noise_g', type=int, default=25)
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/RIDNet/RIDNet.pth')
    args = parser.parse_args()
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]
    test_root = os.path.join(args.test_path, f'X{args.noise_g}')
    result_root = f'results/RIDNet/{os.path.basename(args.test_path)}'
    os.makedirs(result_root, exist_ok=True)

    # set up the RIDNet
    net = RIDNet(3, 64, 3).to(device)
    checkpoint = torch.load(
        args.model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint)
    net.eval()

    # scan all the jpg and png images
    for img_path in sorted(
            glob.glob(os.path.join(test_root, '*.[jp][pn]g'))):
        img_name = os.path.basename(img_path).split('.')[0]
        print(f'Processing {img_name} image ...')
        # read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]],
                                            (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        with torch.no_grad():
            output = net(img)
        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 255).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = output.round().astype(np.uint8)
        save_img_path = f'{result_root}/{img_name}_x{args.noise_g}_RIDNet.png'
        cv2.imwrite(save_img_path, output)
