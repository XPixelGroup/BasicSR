import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def main():
    # Configurations
    # -------------------------------------------------------------------------
    folder_gt = 'datasets/celeba/celeba_512_validation'
    folder_restored = 'datasets/celeba/celeba_512_validation_lq'
    # crop_border = 4
    suffix = ''
    # -------------------------------------------------------------------------
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.
        img_restored = cv2.imread(
            osp.join(folder_restored, basename + suffix + ext),
            cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored],
                                          bgr2rgb=True,
                                          float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(
            img_restored.unsqueeze(0).cuda(),
            img_gt.unsqueeze(0).cuda())

        print('{:3d}: {:25}. \tLPIPS: {:.6f}.'.format(i + 1, basename, lpips_val))
        lpips_all.append(lpips_val)

    print('Average: LPIPS: {:.6f}'.format(sum(lpips_all) / len(lpips_all)))


if __name__ == '__main__':
    main()
