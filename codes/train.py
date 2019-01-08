import os.path
import sys
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging

import torch

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)

    util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old experiment folder if exists
    util.mkdirs((path for key, path in opt['path'].items()
                 if not key == 'experiments_root' and 'pretrain_model' not in key))

    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # config loggers. Before it, the log will not work
    util.setup_logger(None, opt['path']['log'] + '/train.log', level=logging.INFO, screen=True)
    util.setup_logger('val', opt['path']['log'] + '/val.log', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epoches = int(math.ceil(total_iters / train_size))
            logger.info('Total epoches needed: {:d} for iters {:,d}'.format(
                total_epoches, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_dataset_opt = dataset_opt
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                      len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # Create model
    model = create_model(opt)
    epoch_start, current_step  = model.load_checkpoint()
    logger.info('Start training from epoch:{:3d}, iter:{:8d}'.format(epoch_start, current_step))
    for epoch in range(epoch_start, total_epoches):
        for i, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            # update learning rate
            model.update_learning_rate()

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['HR'])  # uint8

                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(\
                        img_name, current_step))
                    util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    crop_size = opt['scale']
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)

                avg_psnr = avg_psnr / idx

                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    epoch, current_step, avg_psnr))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

            # save models
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving the model.')
                model.save_checkpoint(epoch, current_step)

    logger.info('Saving the final model.')
    model.save_checkpoint(total_epoches, current_step)
    logger.info('End of training.')


if __name__ == '__main__':
    main()
