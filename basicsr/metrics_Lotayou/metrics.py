import numpy as np
import os
import torch
from scipy import linalg
from tqdm import tqdm, trange

from basicsr.metrics_Lotayou.metric_util import crop_border_pixels, reorder_image, to_y_channel
from basicsr.utils.img_util import tensor2img
from basicsr.utils.registry import METRIC_REGISTRY_LOTAYOU


def _preliminary_check(fake, real, crop_border, input_order='NCHW', test_y_channel=False, out_range=1.0):
    """ Takes care of checking, reordering, cropping & color space transform """

    if real is not None:
        assert fake.shape == real.shape, (f'Image shapes are differnet: {fake.shape}, {real.shape}.')

    if input_order not in ['NHWC', 'NCHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"NHWC" and "NCHW"')

    fake = reorder_image(fake, input_order=input_order)
    real = reorder_image(real, input_order=input_order)

    if crop_border > 0:
        fake = crop_border_pixels(fake, border_size=crop_border)
        real = crop_border_pixels(real, border_size=crop_border)

    if test_y_channel:
        fake = to_y_channel(fake, out_range)
        real = to_y_channel(real, out_range)
    return fake, real


@METRIC_REGISTRY_LOTAYOU.register()
def calculate_psnr(fake, real, crop_border, input_order='NCHW', test_y_channel=False, device=None):
    """ Calculate PSNR for a batch (Peak Signal-to-Noise Ratio).
        PSNR is simple, no need to convert to GPU.
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        fake, real (torch.tensor): N*C*H*W, detached cpu, [0, 1] range
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
        device (torch.device|None): specifying device to calculate experiments.
            Default: None -> torch.device('cuda')

    Returns:
        float: psnr result.

    # 20210526: Test passed: 32.5297 (old) vs. 32.5405 (new)
        relative error: 3e-4
    """

    print('[Lotayou] Warning: batch_psnr does not support (h,w) gray images for now.')
    if not device:
        device = torch.device('cuda')
    fake, real = _preliminary_check(fake, real, crop_border, input_order, test_y_channel)
    mse = torch.mean((fake - real)**2, dim=(1, 2, 3))
    if mse.max() == 0:
        return float('inf')
    return 20. * -torch.log10(torch.sqrt(mse)).mean()


@METRIC_REGISTRY_LOTAYOU.register()
def calculate_ssim(fake, real, crop_border, input_order='NCHW', test_y_channel=False, chunk_size=100, device=None):
    """Calculate SSIM for a minibatch (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity
    Using the open-source package from: https://github.com/VainF/pytorch-msssim

    Args:
        fake, real (torch.tensor): N*C*H*W, detached cpu, [0, 1] range
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
        chunk_size (int): Minibatch size for testing to avoid memory overflow.
            Default: 20. (For 11GB card)
        device (torch.device|None): specifying device to calculate experiments.
            Default: None -> torch.device('cuda')

    Returns:
        float: ssim result.
    # 20210527: Test passed: 0.8915 (old) vs. 0.8922 (new)
        relative error: 8e-4
        CPU speed (4.09 fps)  GPU speed (~180 fps)
    """

    try:
        from pytorch_msssim import ssim
    except ImportError:
        print('pytorch_msssim not found. Installing via pip...')
        os.system('pip install pytorch_msssim')

    if not device:
        device = torch.device('cuda')
    fake, real = _preliminary_check(fake, real, crop_border, input_order, test_y_channel)
    N = fake.shape[0]
    values = 0.0
    for i in trange(0, N, chunk_size):
        fake_gpu_chunk = fake[i:i + chunk_size].to(device)
        real_gpu_chunk = real[i:i + chunk_size].to(device)
        values += ssim(fake_gpu_chunk, real_gpu_chunk, data_range=1.0, size_average=False).sum()
    return values / N


@METRIC_REGISTRY_LOTAYOU.register()
def calculate_msssim(fake, real, crop_border, input_order='NCHW', test_y_channel=False, chunk_size=100, device=None):
    """Calculate Multiscale SSIM for a minibatch (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity
    Using the open-source package from: https://github.com/VainF/pytorch-msssim

    Args:
        fake, real (torch.tensor): N*C*H*W, detached cpu, [0, 1] range
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
        chunk_size (int): Minibatch size for testing to avoid memory overflow.
            Default: 100. (For 11GB card)
        device (torch.device|None): specifying device to calculate experiments.
            Default: None -> torch.device('cuda')

    Returns:
        float: Multiscale ssim result.
    # 20210527: Test passed: 0.9791 (new) vs. 0.9789 (Face-Renovation repo)
        relative error: 2e-4
        CPU speed (3.25 fps)  GPU speed (~140 fps)
    """
    try:
        from pytorch_msssim import ms_ssim
    except ImportError:
        print('pytorch_msssim not found. Installing via pip...')
        os.system('pip install pytorch_msssim')

    if not device:
        device = torch.device('cuda')
    fake, real = _preliminary_check(fake, real, crop_border, input_order, test_y_channel)
    N = fake.shape[0]
    values = 0.0
    for i in trange(0, N, chunk_size):
        fake_gpu_chunk = fake[i:i + chunk_size].to(device)
        real_gpu_chunk = real[i:i + chunk_size].to(device)
        values += ms_ssim(fake_gpu_chunk, real_gpu_chunk, data_range=1.0, size_average=False).sum()
    return values / N


@METRIC_REGISTRY_LOTAYOU.register()
def calculate_lpips(
    fake,
    real,
    crop_border,
    input_order='NCHW',
    test_y_channel=False,
    chunk_size=20,
    device=None,
    lpips_backbone='alex',
):
    """ Calculate Learned Perceptual Similarity.

    Ref:
    Zhang. et. al, The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018
    Using the open-source package from: https://github.com/richzhang/PerceptualSimilarity

    Args:
        fake, real (torch.tensor): N*C*H*W, detached cpu, [0, 1] range
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
        chunk_size (int): Minibatch size for testing to avoid memory overflow.
            Default: 20. (For 11GB card)
        device (torch.device|None): specifying device to calculate experiments.
            Default: None -> torch.device('cuda')
        lpips_backbone (str): Backbone network for feature extraction. (alex|vgg)
            Default: 'alex'
    Returns:
        float: Multiscale ssim result.
    # 20210527: Test passed: 0.1231 (new) vs 0.1238 (Face-Renovation repo)
        GPU speed (~300 fps)
    """
    try:
        from lpips import LPIPS
    except ImportError:
        print('lpips not found. Installing via pip...')
        os.system('pip install lpips')

    if not device:
        device = torch.device('cuda')
    fake, real = _preliminary_check(fake, real, crop_border, input_order, test_y_channel)
    lpips_model = LPIPS(net=lpips_backbone).to(device)
    N = fake.shape[0]
    values = 0.0
    for i in trange(0, N, chunk_size):
        fake_gpu_chunk = fake[i:i + chunk_size].to(device)
        real_gpu_chunk = real[i:i + chunk_size].to(device)
        value = lpips_model(fake_gpu_chunk, real_gpu_chunk, normalize=True)
        values += value.sum()
    return values / N


@METRIC_REGISTRY_LOTAYOU.register()
def calculate_niqe(
    fake,
    real=None,
    crop_border=0,
    input_order='NCHW',
    convert_to='y',
    num_thread=8,
):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.

    this function does not support GPU

    20210526 [Lotayou]: NIQE is a blind metric that doesn't require reference images.
    To align the parameter with psnr/ssim, we keep a dummy variable.

    Args:
        fake, real (torch.tensor): N*C*H*W, detached cpu, [0, 1] range
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.
        convert_to (str): Color space for conversion (y|gray). Default: 'y'.
            If 'y': Test on Y channel of YCbCr.
            If 'gray': Test using cv2.BGR2GRAY,
        num_thread (int): Number of thread used for metric computation.
            Default: 8 (If set to 1, rollback to single thread)

    Returns:
        float: NIQE result.

    # 20210527: Test passed: 5.9211 (new) vs 5.8586 (basicsr/metrics/niqe.py:calculate_niqe)
        CPU speed (1 thread: 7.44 fps; 6 thread: 21 fps)
    """

    from basicsr.metrics_Lotayou.niqe import niqe

    test_y_channel = (convert_to == 'y')
    # Convert from rgb to bgr first
    inv_index = torch.LongTensor([2, 1, 0])
    fake = fake[:, inv_index]
    fake, real = _preliminary_check(fake, real, crop_border, input_order, test_y_channel, out_range=255.0)
    if convert_to == 'gray':
        # https://docs.opencv.org/4.5.2/de/d25/imgproc_color_conversions.html
        bgr2gray = torch.tensor([0.1140, 0.5870, 0.2989], dtype=fake.dtype, device=fake.device) * 255.
        fake = torch.einsum('nchw,c->nhw', fake, bgr2gray)
    fake = fake.numpy()

    # we use the official params estimated from the pristine dataset.
    niqe_pris_params = np.load('basicsr/metrics/niqe_pris_params.npz')
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']

    if num_thread <= 1:
        total = 0.0
        for _fake in tqdm(fake):
            total += niqe(
                _fake, mu_pris_param=mu_pris_param, cov_pris_param=cov_pris_param,
                gaussian_window=gaussian_window).sum()
        return total / fake.shape[0]
    else:
        from functools import partial
        from multiprocessing import Pool

        partial_niqe = partial(
            niqe, mu_pris_param=mu_pris_param, cov_pris_param=cov_pris_param, gaussian_window=gaussian_window)

        with Pool(num_thread) as p:
            niqe_vals = p.map(partial_niqe, fake)
        fake_val = np.array(niqe_vals).mean()
        return fake_val


@METRIC_REGISTRY_LOTAYOU.register()
def calculate_fid(fake,
                  real,
                  crop_border,
                  input_order='NCHW',
                  test_y_channel=False,
                  chunk_size=50,
                  device=None,
                  inception_feature_level=3,
                  use_bgr_order=True):
    """ Calculate Frechet Inception Distance

    Ref:
    GANs trained by a two time-scale update rule converge to a local nash equilibrium, NIPS 2017
    Using the open-source package from: https://github.com/mseitzer/pytorch-fid

    Args:
        fake, real (torch.tensor): N*C*H*W, detached cpu, [0, 1] range
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
        chunk_size (int): Minibatch size for testing to avoid memory overflow.
            Default: 20. (For 11GB card)
        device (torch.device|None): specifying device to calculate experiments.
            Default: None -> torch.device('cuda')
        inception_feature_level (int): specifying which layer to extract features.
            Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling (Default)
        use_bgr_order (bool): By default BasicSR loads tensor in RGB format
            needs manual conversion for a fair comparison. Default: True
    Returns:
        float: FID distance between fake and real tensors.

    # 20210528: Test Passed: 35.0772 (new) vs 34.9394 (Face-Renovation repo)
        The error is mainly due to uint8 quantization, since Face-Renovation reads in
        generated png images.

    """
    from basicsr.archs.inception import InceptionV3
    if not device:
        device = torch.device('cuda')
    inception_model = InceptionV3([inception_feature_level]).eval().to(device)

    fake, real = _preliminary_check(fake, real, crop_border, input_order, test_y_channel)
    N = fake.shape[0]

    if use_bgr_order:
        inv_index = torch.LongTensor([2, 1, 0])
        fake = fake[:, inv_index]
        real = real[:, inv_index]

    # Extract feature embeddings
    fake_embeddings, real_embeddings = [], []
    with torch.no_grad():
        for i in trange(0, N, chunk_size):
            fake_gpu_chunk = fake[i:i + chunk_size].to(device)
            real_gpu_chunk = real[i:i + chunk_size].to(device)
            # By default inception_model returns a list, only use the last item
            fake_pred = inception_model(fake_gpu_chunk)[0]
            fake_pred = fake_pred.cpu().data.numpy().reshape(fake_pred.shape[0], -1)
            real_pred = inception_model(real_gpu_chunk)[0]
            real_pred = real_pred.cpu().data.numpy().reshape(real_pred.shape[0], -1)
            fake_embeddings.append(fake_pred)
            real_embeddings.append(real_pred)

    fake_embeddings = np.concatenate(fake_embeddings, axis=0)
    real_embeddings = np.concatenate(real_embeddings, axis=0)
    mu_fake = np.mean(fake_embeddings, axis=0)
    sigma_fake = np.cov(fake_embeddings, rowvar=False)
    mu_real = np.mean(real_embeddings, axis=0)
    sigma_real = np.cov(real_embeddings, rowvar=False)
    return _calculate_fid(mu_fake, sigma_fake, mu_real, sigma_real)


# migrated from metrics/fid.py
def _calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Args:
        mu1 (np.array): The sample mean over activations.
        sigma1 (np.array): The covariance matrix over activations for
            generated samples.
        mu2 (np.array): The sample mean over activations, precalculated on an
               representative data set.
        sigma2 (np.array): The covariance matrix over activations,
            precalculated on an representative data set.

    Returns:
        float: The Frechet Distance.
    """
    assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Two covariances have different dimensions'

    cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Product might be almost singular
    if not np.isfinite(cov_sqrt).all():
        print('Product of cov matrices is singular. Adding {eps} to diagonal ' 'of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')
        cov_sqrt = cov_sqrt.real

    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff
    trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace

    return fid


@METRIC_REGISTRY_LOTAYOU.register()
def calculate_fed(fake, real, crop_border, input_order='NCHW', test_y_channel=False):
    """ Calculate feature embedding distance (fed) between paired face images.

    Based on https://github.com/ageitgey/face_recognition
    Note: This metric is CPU-based and does not support multi-threading.
    Furthermore, it sometimes lead to segmentation error (core dumped)
    But it always happens after evaluation, so it's tolerable.

    Args:
        fake, real (torch.tensor): N*C*H*W, detached cpu, [0, 1] range
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: fed result.

    # 20210528: Test passed: 0.0625 (Face-Renovation repo) vs. 0.0625 (new)
    """
    try:
        from face_recognition import face_encodings
    except ImportError:
        print('face_recognition not found. Installing via pip...')
        os.system('pip install face_recognition')

    fake, real = _preliminary_check(fake, real, crop_border, input_order, test_y_channel)
    fake = tensor2img([_item for _item in fake], rgb2bgr=True)
    real = tensor2img([_item for _item in real], rgb2bgr=True)

    values = []
    for _fake, _real in tqdm(list(zip(fake, real))):
        try:
            fake_enc = face_encodings(_fake)[0]
            real_enc = face_encodings(_real)[0]
            enc_dist = np.linalg.norm(fake_enc - real_enc)
            values.append(enc_dist)
        except Exception:
            pass

    if len(values) == 0:
        raise ValueError('None of the testing samples contain detectable faces --- weird:)')
    else:
        return np.array(values).mean()


@METRIC_REGISTRY_LOTAYOU.register()
def calculate_lle(fake, real, crop_border, input_order='NCHW', test_y_channel=False):
    """ Calculate landmark localization error (lle) between paired face images.

    Based on https://github.com/ageitgey/face_recognition
    Note: This metric is CPU-based and does not support multi-threading.
        Furthermore, it sometimes lead to segmentation error (core dumped)
        But it always happens after evaluation, so it's tolerable.

    Args:
        fake, real (torch.tensor): N*C*H*W, detached cpu, [0, 1] range
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: lle result.

    # 20210528: Test passed: 1.8041 (Face-Renovation repo) vs. 1.8041 (new)
        CPU speed (1 thread: ~4.7 fps)
    """

    def _convert_dict(lm_dict):
        """ Convert component-wise landmarks into a vector"""
        lm_list = []
        for k, v in lm_dict.items():
            lm_list += v
        return np.array(lm_list)

    try:
        from face_recognition import face_landmarks
    except ImportError:
        print('face_recognition not found. Installing via pip...')
        os.system('pip install face_recognition')

    fake, real = _preliminary_check(fake, real, crop_border, input_order, test_y_channel)
    fake = tensor2img([_item for _item in fake], rgb2bgr=True)
    real = tensor2img([_item for _item in real], rgb2bgr=True)

    values = []
    for _fake, _real in tqdm(list(zip(fake, real))):
        try:
            fake_lm = _convert_dict(face_landmarks(_fake)[0])
            real_lm = _convert_dict(face_landmarks(_real)[0])
            # result: a dictionary containing landmarks and such
            lm_dist = np.linalg.norm(fake_lm - real_lm, axis=1).mean()
            values.append(lm_dist)
        except Exception:
            pass

    if len(values) == 0:
        raise ValueError('None of the testing samples contain detectable faces --- weird:)')
    else:
        return np.array(values).mean()
