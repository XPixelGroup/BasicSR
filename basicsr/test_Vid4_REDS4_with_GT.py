'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression)
datasets
'''

import glob
import logging
import os
import os.path as osp

import cv2
import numpy as np
import torch

import basicsr.data.util as data_util
import basicsr.models.archs.EDVR_arch as EDVR_arch
import basicsr.utils.util as util


# TODO: modify it
def single_forward(model, inp):
    """PyTorch model forward (single test), it is just a simple warpper
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    with torch.no_grad():
        model_output = model(inp)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output


# TODO: modify it
def flipx4_forward(model, inp):
    """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W,
    flip H and W.

    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    # normal
    output_f = single_forward(model, inp)

    # flip W
    output = single_forward(model, torch.flip(inp, (-1, )))
    output_f = output_f + torch.flip(output, (-1, ))
    # flip H
    output = single_forward(model, torch.flip(inp, (-2, )))
    output_f = output_f + torch.flip(output, (-2, ))
    # flip both H and W
    output = single_forward(model, torch.flip(inp, (-2, -1)))
    output_f = output_f + torch.flip(output, (-2, -1))

    return output_f / 4


def main():
    #####
    # configurations
    #####
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_mode = 'Vid4'  # Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
    # Vid4: SR
    # REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
    #        blur (deblur-clean), blur_comp (deblur-compression).
    stage = 1  # 1 or 2, use two stage strategy for REDS dataset.
    flip_test = False
    ###################
    # model
    if data_mode == 'Vid4':
        if stage == 1:
            model_path = '../experiments/pretrained_models/EDVR_Vimeo90K_SR_L.pth'  # noqa: E501
        else:
            raise ValueError('Vid4 does not support stage 2.')
    elif data_mode == 'sharp_bicubic':
        if stage == 1:
            model_path = '../experiments/pretrained_models/EDVR_REDS_SR_L.pth'
        else:
            model_path = '../experiments/pretrained_models/EDVR_REDS_SR_Stage2.pth'  # noqa: E501
    elif data_mode == 'blur_bicubic':
        if stage == 1:
            model_path = '../experiments/pretrained_models/EDVR_REDS_SRblur_L.pth'  # noqa: E501
        else:
            model_path = '../experiments/pretrained_models/EDVR_REDS_SRblur_Stage2.pth'  # noqa: E501
    elif data_mode == 'blur':
        if stage == 1:
            model_path = '../experiments/pretrained_models/EDVR_REDS_deblur_L.pth'  # noqa: E501
        else:
            model_path = '../experiments/pretrained_models/EDVR_REDS_deblur_Stage2.pth'  # noqa: E501
    elif data_mode == 'blur_comp':
        if stage == 1:
            model_path = '../experiments/pretrained_models/EDVR_REDS_deblurcomp_L.pth'  # noqa: E501
        else:
            model_path = '../experiments/pretrained_models/EDVR_REDS_deblurcomp_Stage2.pth'  # noqa: E501
    else:
        raise NotImplementedError

    if data_mode == 'Vid4':
        N_in = 7  # use N_in images to restore one HR image
    else:
        N_in = 5

    predeblur, hr_in = False, False
    num_reconstruct_block = 40
    if data_mode == 'blur_bicubic':
        predeblur = True
    if data_mode == 'blur' or data_mode == 'blur_comp':
        predeblur, hr_in = True, True
    if stage == 2:
        hr_in = True
        num_reconstruct_block = 20
    model = EDVR_arch.EDVR(
        128,
        N_in,
        8,
        5,
        num_reconstruct_block,
        predeblur=predeblur,
        hr_in=hr_in)

    # dataset
    if data_mode == 'Vid4':
        test_dataset_folder = '../datasets/Vid4/BIx4'
        gt_dataset_folder = '../datasets/Vid4/GT'
    else:
        if stage == 1:
            test_dataset_folder = '../datasets/REDS4/{}'.format(data_mode)
        else:
            test_dataset_folder = '../results/REDS-EDVR_REDS_SR_L_flipx4'
            print('You should modify the test_dataset_folder path for stage 2')
        gt_dataset_folder = '../datasets/REDS4/GT'

    # evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    if data_mode == 'Vid4' or data_mode == 'sharp_bicubic':
        padding = 'new_info'
    else:
        padding = 'replicate'
    save_imgs = True

    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    util.setup_logger(
        'base',
        save_folder,
        'test',
        level=logging.INFO,
        screen=True,
        tofile=True)
    logger = logging.getLogger('base')

    # log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    # set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    subfolder_gt_l = sorted(glob.glob(osp.join(gt_dataset_folder, '*')))
    # for each subfolder
    for subfolder, subfolder_gt in zip(subfolder_l, subfolder_gt_l):
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        # read lq and gt images
        imgs_lq = data_util.read_img_seq(subfolder)
        img_gt_l = []
        for img_gt_path in sorted(glob.glob(osp.join(subfolder_gt, '*'))):
            img_gt_l.append(data_util.read_img(None, img_gt_path))

        avg_psnr, avg_psnr_border, avg_psnr_center = 0, 0, 0
        N_border, N_center = 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.generate_frame_indices(
                img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_lq.index_select(
                0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            if flip_test:
                output = util.flipx4_forward(model, imgs_in)
            else:
                output = util.single_forward(model, imgs_in)
            output = util.tensor2img(output.squeeze(0))

            if save_imgs:
                cv2.imwrite(
                    osp.join(save_subfolder, '{}.png'.format(img_name)),
                    output)

            # calculate PSNR
            output = output / 255.
            gt = np.copy(img_gt_l[img_idx])
            # For REDS, evaluate on RGB channels; for Vid4, evaluate on the
            # Y channel
            if data_mode == 'Vid4':  # bgr2y, [0, 1]
                gt = data_util.bgr2ycbcr(gt, only_y=True)
                output = data_util.bgr2ycbcr(output, only_y=True)

            output, gt = util.crop_border([output, gt], crop_border)
            crt_psnr = util.calculate_psnr(output * 255, gt * 255)
            logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB'.format(
                img_idx + 1, img_name, crt_psnr))

            if img_idx >= border_frame and img_idx < max_idx - border_frame:
                # center frames
                avg_psnr_center += crt_psnr
                N_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                N_border += 1

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)

        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                    'Center PSNR: {:.6f} dB for {} frames; '
                    'Border PSNR: {:.6f} dB for {} frames.'.format(
                        subfolder_name, avg_psnr, (N_center + N_border),
                        avg_psnr_center, N_center, avg_psnr_border, N_border))

    logger.info('#### Tidy Outputs ####')
    for subfolder_name, psnr, psnr_center, psnr_border in zip(
            subfolder_name_l, avg_psnr_l, avg_psnr_center_l,
            avg_psnr_border_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                    'Center PSNR: {:.6f} dB. '
                    'Border PSNR: {:.6f} dB.'.format(subfolder_name, psnr,
                                                     psnr_center, psnr_border))
    logger.info('#### Final Results ####')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
                    sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
                    sum(avg_psnr_center_l) / len(avg_psnr_center_l),
                    sum(avg_psnr_border_l) / len(avg_psnr_border_l)))


if __name__ == '__main__':
    main()
