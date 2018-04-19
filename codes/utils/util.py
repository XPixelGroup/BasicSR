import os
import math
import numpy as np
from datetime import datetime
from PIL import Image

import torch
from torchvision.utils import make_grid


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def save_img_np(img_np, img_path, mode='RGB'):
    img_pil = Image.fromarray(img_np, mode=mode)
    img_pil.save(img_path)

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
        new_name = path + '_archivedAt_' + get_timestamp()
        print('Path already exists. Rename it to [%s]' % new_name)
        os.rename(path, new_name)
    os.makedirs(path)


"""
Converts a Tensor into an image Numpy array
Input should be either in 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W)
If input is in 4D, it is splited along the first dimension to provide grid view.
Otherwise, the tensor is assume to be single image.
Input type: float [-1, 1] (default)
Output type: np.uint8 [0,255] (default)
Output dim: 3D(H,W,C) (for 4D and 3D input) or 2D(H,W) (for 2D input)
"""
def tensor2img_np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)  # Clamp is for on hard_tanh
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
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
            'Only support 4D, 3D and 2D tensor. But receieved tensor with dimension = %d' % n_dim)
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round(
        )  # This is important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def rgb2gray(img):
    in_img_type = img.dtype
    img.astype(np.float64)
    img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).round()
    return img_gray.astype(in_img_type)


def rgb2y(img):
    assert (img.dtype == np.uint8)
    in_img_type = img.dtype
    img.astype(np.float64)
    img_y = ((np.dot(img[..., :3], [65.481, 128.553, 24.966])) / 255.0 + 16.0).round()
    return img_y.astype(in_img_type)
