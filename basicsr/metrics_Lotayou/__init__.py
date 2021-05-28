from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY_LOTAYOU
from .metrics import (calculate_fed, calculate_fid, calculate_lle, calculate_lpips, calculate_msssim, calculate_niqe,
                      calculate_psnr, calculate_ssim)

__all__ = [
    'calculate_fed', 'calculate_fid', 'calculate_lle', 'calculate_lpips', 'calculate_msssim', 'calculate_niqe',
    'calculate_psnr', 'calculate_ssim'
]


def calculate_metric(fake_tensors, real_tensors, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must constain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY_LOTAYOU.get(metric_type)(fake_tensors, real_tensors, **opt)
    return metric
