import numpy as np
import math
from skimage.measure import compare_ssim

"""
img1, img2 should be in numpy format with type uint8.
"""

def psnr(img1, img2):
    assert (img1.dtype == img2.dtype == np.uint8)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2, multichannel=False):
    assert (img1.dtype == img2.dtype == np.uint8)
    return compare_ssim(img1,img2,multichannel=multichannel)