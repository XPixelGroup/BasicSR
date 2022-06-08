import cv2
import glob
import logging
import os
import os.path as osp

from basicsr.metrics import psnr_ssim
from basicsr.utils import get_root_logger, get_time_str


def main():

    sr_folder = 'results/BasicVSRPP'
    gt_folder = 'datasets/REDS4/GT'

    # logger
    log_file = osp.join(sr_folder, f'psnr_test_{get_time_str()}.log')
    logger = get_root_logger(logger_name='bascivsrpp', log_level=logging.INFO, log_file=log_file)

    avg_psnr_l = []

    subfolder_sr_l = sorted(glob.glob(osp.join(sr_folder, '*')))
    subfolder_gt_l = sorted(glob.glob(osp.join(gt_folder, '*')))

    # for each subfolder
    subfolder_names = []
    for subfolder_sr, subfolder_gt in zip(subfolder_sr_l, subfolder_gt_l):
        subfolder_name = osp.basename(subfolder_sr)
        subfolder_names.append(subfolder_name)

        avg_psnr = 0
        name_idx = 0
        img_name_list = sorted(os.listdir(subfolder_gt))
        for img_name in img_name_list:
            img_basename = os.path.splitext(img_name)[0]
            # read SR image and GT image
            img_sr = cv2.imread(osp.join(subfolder_sr, f'{img_basename}_BasicVSRPP.png'), cv2.IMREAD_UNCHANGED)
            # read GT image
            img_gt = cv2.imread(osp.join(subfolder_gt, f'{img_basename}.png'), cv2.IMREAD_UNCHANGED)
            crt_psnr = psnr_ssim.calculate_psnr(img_sr, img_gt, crop_border=0, test_y_channel=False)

            avg_psnr += crt_psnr
            logger.info(f'{subfolder_name}--{img_name} - PSNR: {crt_psnr:.6f} dB. ')
            name_idx += 1

        avg_psnr /= name_idx
        avg_psnr_l.append(avg_psnr)

    for folder_idx, subfolder_name in enumerate(subfolder_names):
        logger.info(f'Folder {subfolder_name} - Average PSNR: {avg_psnr_l[folder_idx]:.6f} dB. ')

    logger.info(f'Average PSNR: {sum(avg_psnr_l) / len(avg_psnr_l):.6f} dB ' f'for {len(subfolder_sr_l)} clips. ')


if __name__ == '__main__':

    main()
