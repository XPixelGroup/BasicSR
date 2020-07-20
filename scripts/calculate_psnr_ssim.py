import mmcv
import numpy as np
from os import path as osp

from basicsr.metrics import calculate_psnr, calculate_ssim


def main():
    """Calculate PSNR and SSIM for images.

    Configurations:
        folder_gt (str): Path to gt (Ground-Truth).
        folder_restored (str): Path to restored images.
        crop_border (int): Crop border for each side.
        suffix (str): Suffix for restored images.
        test_y_channel (bool): If True, test Y channel (In MatLab YCbCr format)
            If False, test RGB channels.
    """
    # Configurations
    # -------------------------------------------------------------------------
    folder_gt = 'datasets/val_set14/Set14'
    folder_restored = 'results/exp/visualization/val_set14'
    crop_border = 4
    suffix = '_expname'
    test_y_channel = False
    # -------------------------------------------------------------------------

    psnr_all = []
    ssim_all = []
    img_list = sorted(mmcv.scandir(folder_gt, recursive=True))

    if test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = mmcv.imread(
            osp.join(folder_gt, img_path), flag='unchanged').astype(
                np.float32) / 255.
        img_restored = mmcv.imread(
            osp.join(folder_restored, basename + suffix + ext),
            flag='unchanged').astype(np.float32) / 255.

        if test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = mmcv.bgr2ycbcr(img_gt, y_only=True)
            img_restored = mmcv.bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
        psnr = calculate_psnr(
            img_gt * 255,
            img_restored * 255,
            crop_border=crop_border,
            input_order='HWC')
        ssim = calculate_ssim(
            img_gt * 255,
            img_restored * 255,
            crop_border=crop_border,
            input_order='HWC')
        print(f'{i+1:3d}: {basename:25}. \tPSNR: {psnr:.6f} dB, '
              f'\tSSIM: {ssim:.6f}')
        psnr_all.append(psnr)
        ssim_all.append(ssim)
    print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, '
          f'SSIM: {sum(ssim_all) / len(ssim_all):.6f}')


if __name__ == '__main__':
    main()
