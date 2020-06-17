import math

import mmcv
import torchvision.utils

from basicsr.data import create_dataloader, create_dataset


def main(mode='folder'):
    """Test paired image dataset.

    Args:
        mode: There are three modes: 'lmdb', 'folder', 'ann_file'.
    """
    opt = {}
    opt['dist'] = False
    opt['gpu_ids'] = [0]
    opt['phase'] = 'train'

    opt['name'] = 'DIV2K'
    opt['type'] = 'PairedImageDataset'
    if mode == 'folder':
        opt['dataroot_gt'] = 'datasets/DIV2K/DIV2K_train_HR_sub'
        opt['dataroot_lq'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
        opt['filename_tmpl'] = '{}'
        opt['io_backend'] = dict(type='disk')
    elif mode == 'ann_file':
        opt['dataroot_gt'] = 'datasets/DIV2K/DIV2K_train_HR_sub'
        opt['dataroot_lq'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
        opt['ann_file'] = 'basicsr/data/meta_info/meta_info_DIV2K800sub_GT.txt'
        opt['filename_tmpl'] = '{}'
        opt['io_backend'] = dict(type='disk')
    elif mode == 'lmdb':
        opt['dataroot_gt'] = 'datasets/DIV2K/DIV2K_train_HR_sub.lmdb'
        opt['dataroot_lq'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb'
        opt['io_backend'] = dict(type='lmdb')

    opt['gt_size'] = 128
    opt['use_flip'] = True
    opt['use_rot'] = True

    opt['use_shuffle'] = True
    opt['num_worker'] = 1
    opt['batch_size'] = 16
    opt['scale'] = 4

    opt['dataset_enlarge_ratio'] = 1

    mmcv.mkdir_or_exist('tmp')

    dataset = create_dataset(opt)
    data_loader = create_dataloader(dataset, opt, opt, None)

    nrow = int(math.sqrt(opt['batch_size']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(i)

        lq = data['lq']
        gt = data['gt']
        lq_path = data['lq_path']
        gt_path = data['gt_path']
        print(lq_path, gt_path)
        torchvision.utils.save_image(
            lq,
            f'tmp/lq_{i:03d}.png',
            nrow=nrow,
            padding=padding,
            normalize=False)
        torchvision.utils.save_image(
            gt,
            f'tmp/gt_{i:03d}.png',
            nrow=nrow,
            padding=padding,
            normalize=False)


if __name__ == '__main__':
    main()
