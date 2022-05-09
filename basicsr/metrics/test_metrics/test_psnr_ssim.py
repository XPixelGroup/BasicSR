import cv2
import torch

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
from basicsr.utils import img2tensor


def test(img_path, img_path2, crop_border, test_y_channel=False):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img_path2, cv2.IMREAD_UNCHANGED)

    # --------------------- Numpy ---------------------
    psnr = calculate_psnr(img, img2, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
    ssim = calculate_ssim(img, img2, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
    print(f'\tNumpy\tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}')

    # --------------------- PyTorch (CPU) ---------------------
    img = img2tensor(img / 255., bgr2rgb=True, float32=True).unsqueeze_(0)
    img2 = img2tensor(img2 / 255., bgr2rgb=True, float32=True).unsqueeze_(0)

    psnr_pth = calculate_psnr_pt(img, img2, crop_border=crop_border, test_y_channel=test_y_channel)
    ssim_pth = calculate_ssim_pt(img, img2, crop_border=crop_border, test_y_channel=test_y_channel)
    print(f'\tTensor (CPU) \tPSNR: {psnr_pth[0]:.6f} dB, \tSSIM: {ssim_pth[0]:.6f}')

    # --------------------- PyTorch (GPU) ---------------------
    img = img.cuda()
    img2 = img2.cuda()
    psnr_pth = calculate_psnr_pt(img, img2, crop_border=crop_border, test_y_channel=test_y_channel)
    ssim_pth = calculate_ssim_pt(img, img2, crop_border=crop_border, test_y_channel=test_y_channel)
    print(f'\tTensor (GPU) \tPSNR: {psnr_pth[0]:.6f} dB, \tSSIM: {ssim_pth[0]:.6f}')

    psnr_pth = calculate_psnr_pt(
        torch.repeat_interleave(img, 2, dim=0),
        torch.repeat_interleave(img2, 2, dim=0),
        crop_border=crop_border,
        test_y_channel=test_y_channel)
    ssim_pth = calculate_ssim_pt(
        torch.repeat_interleave(img, 2, dim=0),
        torch.repeat_interleave(img2, 2, dim=0),
        crop_border=crop_border,
        test_y_channel=test_y_channel)
    print(f'\tTensor (GPU batch) \tPSNR: {psnr_pth[0]:.6f}, {psnr_pth[1]:.6f} dB,'
          f'\tSSIM: {ssim_pth[0]:.6f}, {ssim_pth[1]:.6f}')


if __name__ == '__main__':
    test('tests/data/bic/baboon.png', 'tests/data/gt/baboon.png', crop_border=4, test_y_channel=False)
    test('tests/data/bic/baboon.png', 'tests/data/gt/baboon.png', crop_border=4, test_y_channel=True)

    test('tests/data/bic/comic.png', 'tests/data/gt/comic.png', crop_border=4, test_y_channel=False)
    test('tests/data/bic/comic.png', 'tests/data/gt/comic.png', crop_border=4, test_y_channel=True)
