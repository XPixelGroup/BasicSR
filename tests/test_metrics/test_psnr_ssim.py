import numpy as np
import pytest

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


def test_calculate_psnr():
    """Test metric: calculate_psnr"""

    # mismatched image shapes
    with pytest.raises(AssertionError):
        calculate_psnr(np.ones((16, 16)), np.ones((10, 10)), crop_border=0)

    # wrong input order
    with pytest.raises(ValueError):
        calculate_psnr(np.ones((16, 16)), np.ones((16, 16)), crop_border=1, input_order='WRONG')

    out = calculate_psnr(np.ones((10, 10, 3)), np.ones((10, 10, 3)) * 2, crop_border=1, test_y_channel=True)
    assert isinstance(out, float)

    # test float inf
    out = calculate_psnr(np.ones((10, 10, 3)), np.ones((10, 10, 3)), crop_border=0)
    assert out == float('inf')


def test_calculate_ssim():
    """Test metric: calculate_ssim"""

    # mismatched image shapes
    with pytest.raises(AssertionError):
        calculate_ssim(np.ones((16, 16)), np.ones((10, 10)), crop_border=0)

    # wrong input order
    with pytest.raises(ValueError):
        calculate_ssim(np.ones((16, 16)), np.ones((16, 16)), crop_border=1, input_order='WRONG')

    out = calculate_ssim(np.ones((10, 10, 3)), np.ones((10, 10, 3)) * 2, crop_border=1, test_y_channel=True)
    assert isinstance(out, float)
