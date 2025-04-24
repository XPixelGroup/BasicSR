# img_process_util.py

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)

    Returns:
        Tensor: Filtered image with shape (b, c, h, w)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.

    Args:
        img (Numpy array or Tensor): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int): Threshold for the sharpening mask.

    Returns:
        Numpy array or Tensor: Sharpened image, same type as input
    """
    if radius % 2 == 0:
        radius += 1

    if torch.is_tensor(img):
        device = img.device
        # Move to CPU for OpenCV operations
        img_np = img.cpu().numpy()
    else:
        device = None
        img_np = img

    blur = cv2.GaussianBlur(img_np, (radius, radius), 0)
    residual = img_np - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img_np + weight * residual
    sharp = np.clip(sharp, 0, 1)
    output = soft_mask * sharp + (1 - soft_mask) * img_np

    if device is not None:
        # Convert back to tensor if input was tensor
        output = torch.from_numpy(output).to(device)
    return output


class USMSharp(torch.nn.Module):
    """PyTorch version of Unsharp Masking sharpening layer."""

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        """Forward function.

        Args:
            img (Tensor): Input image tensor (b, c, h, w)
            weight (float): Sharpening weight
            threshold (float): Threshold for the sharpening mask

        Returns:
            Tensor: Sharpened image tensor (b, c, h, w)
        """
        device = img.device
        self.kernel = self.kernel.to(device)  # Ensure kernel is on same device as input

        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clamp(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img