import argparse
import cv2
import os
import warnings

from basicsr.metrics import calculate_niqe
from basicsr.utils import scandir


def main(args):

    niqe_all = []
    img_list = sorted(scandir(args.input, recursive=True, full_path=True))

    for i, img_path in enumerate(img_list):
        basename, _ = os.path.splitext(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            niqe_score = calculate_niqe(img, args.crop_border, input_order='HWC', convert_to='y')
        print(f'{i+1:3d}: {basename:25}. \tNIQE: {niqe_score:.6f}')
        niqe_all.append(niqe_score)

    print(args.input)
    print(f'Average: NIQE: {sum(niqe_all) / len(niqe_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/val_set14/Set14', help='Input path')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    args = parser.parse_args()
    main(args)
