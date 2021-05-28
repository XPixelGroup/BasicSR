import torch

# All functions are safe if input img is None (NIQE real)


def crop_border_pixels(img, border_size=4):
    """ Crop border pixels for batched tensors """
    if img is None:
        return None
    if border_size > 0:
        if img.ndim == 4:
            return img[:, :, border_size:-border_size, border_size:-border_size]
        elif img.ndim == 3:
            return img[:, border_size:-border_size, border_size:-border_size]
        else:
            raise (ValueError(f'img must be 3 or 4 dim, got {img.ndim}.'))


def reorder_image(img, input_order='NCHW'):
    """Reorder images to 'NCHW' order.

    If the input_order is (n, h, w), return (n, 1, h, w).
    If the input_order is (n, c, h, w), return as it is.
    If the input_order is (n, h, w, c), return (n, h, w, c).

    Args:
        img (torch.tensor): Input image.
        input_order (str): Whether the input order is 'NCHW' or 'NHWC'.
            If the input image shape is (n, h, w), input_order will not have
            effects. Default: 'NCHW'.

    Returns:
        4-dimension torch.tensor in NCHW format.
    """
    if img is None:
        return None
    if input_order not in ['NHWC', 'NCHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'NHWC' and 'NCHW'")
    if len(img.shape) == 3:
        img.unsqueeze_(dim=1)
    if input_order == 'NHWC':
        img = torch.permute(0, 3, 1, 2)
    return img


def to_y_channel(img, out_range=1.0):
    """Change to Y channel of YCbCr.

    Args:
        img (torch.tensor): 4-dimension torch.float32 tensor in NCHW format. range[0,1]
            If C=1 (gray image), return as it is.
        out_range (float): output maximum, 1.0 or 255. Default: 1.0

    Returns:
        (torch.tensor): 3-dimension torch.float32 tensor in NHW format.
    """
    if img is None:
        return None
    img = img.to(torch.float32)
    if img.ndim == 4 and img.shape[1] == 3:
        img = bgr2ycbcr(img, y_only=True, out_range=out_range)
    return img


def bgr2ycbcr(img, y_only=False, out_range=255.0):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (torch.tensor): 4-dimension torch.float32 tensor in NCHW format. range[0,1]
        y_only (bool): Whether to only return Y channel. Default: False.
        out_range (int|float): output maximum, 1.0 or 255. Default: 255.0

    Returns:
        (torch.tensor): The converted YCbCr image.

    Warning: BasicSR reads images with cv2.imread in BGR format
        but switch to RGB when converting to torch tensors
        so the value order should change accordingly

    """
    # TODO: convert from np to torch

    if y_only:
        out_img = torch.einsum('nchw,c->nhw', img, torch.tensor([24.966, 128.553, 65.481])) + 16.0
    else:
        offset = torch.tensor([16.0, 128.0, 128.0], dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
        out_img = torch.einsum(
            'nchw,cd->ndhw', img,
            torch.tensor([[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]])) + offset

    return out_img * out_range / 255.
