import os
import math
import numpy as np
from datetime import datetime
from PIL import Image
from skimage.measure import compare_ssim

import torch
from torchvision.utils import make_grid


####################
# miscellaneous
####################

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [%s]' % new_name)
        os.rename(path, new_name)
    os.makedirs(path)


####################
# image convert
####################

def tensor2img_np(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But receieved tensor with dimension: %d' % n_dim)
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img_np(img_np, img_path, mode='RGB'):
    img_pil = Image.fromarray(img_np, mode=mode)
    img_pil.save(img_path)


def rgb2gray(img):
    '''
    rgb2gray is the Y component of YUV;
    the same as matlab rgb2gray where coefficients are [0.2989, 0.587, 0.114]
    Input: image Numpy array, [0,255], HWC, RGB
    Output: image Numpy array, [0, 255], HW
    '''
    assert img.dtype == np.uint8, 'np.uint8 is supposed. But received img dtype: %s.' % img.dtype
    in_img_type = img.dtype
    img.astype(np.float64)
    img_gray = np.dot(img[..., :3], [0.2989, 0.587, 0.114]).round()
    return img_gray.astype(in_img_type)


def rgb2ycbcr(img, only_y=True):
    # the same as matlab rgb2ycbcr
    # TODO support double [0, 1]
    assert img.dtype == np.uint8, 'np.uint8 is supposed. But received img dtype: %s.' % img.dtype
    in_img_type = img.dtype
    img.astype(np.float64)
    if only_y: # only return Y channel
        img_y = (np.dot(img[..., :3], [65.481, 128.553, 24.966]) / 255.0 + 16.0).round()
        return img_y.astype(in_img_type)
    else:
        img_ycbcr = (np.matmul(img[..., :3], [[65.481, -37.797, 112.0], \
        [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]).round()
        return img_ycbcr.astype(in_img_type)


def ycbcr2rgb(img):
    # the same as matlab ycbcr2rgb
    # TODO support double [0, 1]
    assert img.dtype == np.uint8, 'np.uint8 is supposed. But received img dtype: %s.' % img.dtype
    in_img_type = img.dtype
    img.astype(np.float64)
    img_rgb = (np.matmul(img[..., :3], [[0.00456621, 0.00456621, 0.00456621], \
        [0, -0.00153632, 0.00791071], [0.00625893, -0.00318811, 0]]) * 255.0 + \
        [-222.921, 135.576, -276.836]).round()
    return img_rgb.astype(in_img_type)


####################
# metric
####################

def psnr(img1, img2):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2, multichannel=False):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    return compare_ssim(img1, img2, multichannel=multichannel)


# test
if __name__ == '__main__':
    import cv2
    img_test = cv2.imread('butterfly.png', cv2.IMREAD_UNCHANGED)
    img_test = img_test[:, :, [2, 1, 0]]
    # rgb2gray
    img_gray = rgb2gray(img_test)
    save_img_np(img_gray, 'test_gray.png', mode='L')
    # rgb2ycbcr
    img_y = rgb2ycbcr(img_test)
    save_img_np(img_y, 'test_y.png', mode='L')
    img_ycbcr = rgb2ycbcr(img_test, only_y=False)
    import scipy.io as sio
    sio.savemat('ycbcr.mat', {'ycbcr': img_ycbcr})
    img_rgb = ycbcr2rgb(img_ycbcr)
    save_img_np(img_rgb, 'test_rgb.png', mode='RGB')
