import cv2
import math
import numpy as np
import os
import torch
from enum import IntEnum, auto
from functools import partial
from torchvision.utils import make_grid

from .color_util import bgr2ycbcr, ycbcr2bgr


class ColorSpace(IntEnum):
    RAW = auto()  # do not convert colorspace
    BGR = auto()
    RGB = auto()
    GRAY = auto()  # YUVJ
    XYZ = auto()
    YCrCb = auto()
    HSV = auto()
    Lab = auto()
    Luv = auto()
    HLS = auto()
    YUV = auto()  # YUVJ
    YUVI420 = auto()  # YUVJ
    Y = GRAY_BT601 = auto()
    YUV_BT601 = auto()
    GRAY_BT709 = auto()
    YUV_BT709 = auto()


BGR2COLOR = {
    ColorSpace.RAW: lambda x: x,
    ColorSpace.BGR: lambda x: x,
    ColorSpace.RGB: partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB),
    ColorSpace.GRAY: partial(cv2.cvtColor, code=cv2.COLOR_BGR2GRAY),
    ColorSpace.XYZ: partial(cv2.cvtColor, code=cv2.COLOR_BGR2XYZ),
    ColorSpace.YCrCb: partial(cv2.cvtColor, code=cv2.COLOR_BGR2YCrCb),
    ColorSpace.HSV: partial(cv2.cvtColor, code=cv2.COLOR_BGR2HSV),
    ColorSpace.Lab: partial(cv2.cvtColor, code=cv2.COLOR_BGR2Lab),
    ColorSpace.Luv: partial(cv2.cvtColor, code=cv2.COLOR_BGR2Luv),
    ColorSpace.HLS: partial(cv2.cvtColor, code=cv2.COLOR_BGR2HLS),
    ColorSpace.YUV: partial(cv2.cvtColor, code=cv2.COLOR_BGR2YUV),
    ColorSpace.YUVI420: partial(cv2.cvtColor, code=cv2.COLOR_BGR2YUV_I420),
    ColorSpace.GRAY_BT601: partial(bgr2ycbcr, y_only=True),
    ColorSpace.YUV_BT601: partial(bgr2ycbcr, y_only=False),
}

COLOR2BGR = {
    ColorSpace.RAW: lambda x: x,
    ColorSpace.BGR: lambda x: x,
    ColorSpace.RGB: partial(cv2.cvtColor, code=cv2.COLOR_RGB2BGR),
    ColorSpace.GRAY: partial(cv2.cvtColor, code=cv2.COLOR_GRAY2BGR),
    ColorSpace.XYZ: partial(cv2.cvtColor, code=cv2.COLOR_XYZ2BGR),
    ColorSpace.YCrCb: partial(cv2.cvtColor, code=cv2.COLOR_YCrCb2BGR),
    ColorSpace.HSV: partial(cv2.cvtColor, code=cv2.COLOR_HSV2BGR),
    ColorSpace.Lab: partial(cv2.cvtColor, code=cv2.COLOR_Lab2BGR),
    ColorSpace.Luv: partial(cv2.cvtColor, code=cv2.COLOR_Luv2BGR),
    ColorSpace.HLS: partial(cv2.cvtColor, code=cv2.COLOR_HLS2BGR),
    ColorSpace.YUV: partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR),
    ColorSpace.YUVI420: partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_I420),
    ColorSpace.GRAY_BT601: ycbcr2bgr,
    ColorSpace.YUV_BT601: ycbcr2bgr,
}


def img2tensor(imgs, color_space=ColorSpace.RGB, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images, MUST be BGR
        color_space (ColorSpace): Target color space of images.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, color_space, float32):
        if img.shape[2] == 3:  # input is bgr
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = BGR2COLOR[color_space](img)
        if img.ndim == 3:  # HWC to CHW
            img = torch.from_numpy(img.transpose(2, 0, 1))
        elif img.ndim == 2:
            img = torch.from_numpy(np.expand_dims(img, axis=0))
        else:
            raise ValueError(f'Unsupported image dim {img.ndim}!')
        if float32:
            img = img.float()
        return img

    if isinstance(color_space, str):
        for cs in ColorSpace:
            if color_space.lower() == cs.name.lower():
                color_space = cs
                break
        if isinstance(color_space, str):
            raise ValueError(f'Do not support color space {color_space} yet!')
    if isinstance(imgs, list):
        return [_totensor(img, color_space, float32) for img in imgs]
    else:
        return _totensor(imgs, color_space, float32)


def tensor2img(tensor, color_space=ColorSpace.RGB, out_type=np.uint8, min_max=None):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        color_space (ColorSpace): Color space of input tensor.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if isinstance(color_space, str):
        for cs in ColorSpace:
            if color_space.lower() == cs.name.lower():
                color_space = cs
                break
        if isinstance(color_space, str):
            raise ValueError(f'Do not support color space {color_space} yet!')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu()

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)  # CHW to HWC
            img_np = COLOR2BGR[color_space](img_np)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)  # CHW to HWC
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                img_np = COLOR2BGR[color_space](img_np)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        # clip BGR to (0, 1)
        img_np = np.clip(img_np, 0, 1)
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def tensor2img_fast(tensor, color_space=ColorSpace.RGB, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        color_space (ColorSpace): Color space of tesnor. Default: ColorSpace.RGB.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    output = COLOR2BGR[color_space](output)
    return output


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]
