import argparse
import cv2
import glob
import numpy as np
import os

from basicsr.utils.matlab_functions import imresize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/REDS4/GT/000', help='input image floder')
    parser.add_argument('--output', type=str, default='datasets/LRbicx2', help='output LRbicx2 folder')
    parser.add_argument('--scale', type=float, default=2, help='upsampling scale')
    parser.add_argument('--mode', type=str, default='up', help='up/down for upsampling/downsampling')
    args = parser.parse_args()

    scale = args.scale
    if args.mode == 'down':
        scale = 1 / args.scale

    os.makedirs(args.output, exist_ok=True)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('processing ', idx, imgname)
        img = cv2.imread(path).astype(np.float32) / 255.
        output = imresize(img, scale)
        output = np.clip((output * 255).round(), 0, 255)
        cv2.imwrite(os.path.join(args.output, f'{imgname}_LRbicx_{args.mode}{args.scale}X.png'), output)


if __name__ == '__main__':
    main()
