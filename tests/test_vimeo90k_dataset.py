import math
import mmcv
import torchvision.utils

from basicsr.data import create_dataloader, create_dataset


def main(mode='folder'):
    """Test vimeo90k dataset.

    Args:
        mode: There are two modes: 'lmdb', 'folder'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'Vimeo90K'
    opt['type'] = 'Vimeo90KDataset'
    if mode == 'folder':
        opt['dataroot_gt'] = 'datasets/vimeo90k/vimeo_septuplet/sequences'
        opt['dataroot_lq'] = 'datasets/vimeo90k/vimeo_septuplet_matlabLRx4/sequences'  # noqa E501
        opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt'  # noqa E501
        opt['io_backend'] = dict(type='disk')
    elif mode == 'lmdb':
        opt['dataroot_gt'] = 'datasets/vimeo90k/vimeo90k_train_GT_only4th.lmdb'
        opt['dataroot_lq'] = 'datasets/vimeo90k/vimeo90k_train_LR7frames.lmdb'
        opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt'  # noqa E501
        opt['io_backend'] = dict(type='lmdb')

    opt['num_frame'] = 7
    opt['gt_size'] = 256
    opt['random_reverse'] = True
    opt['use_flip'] = True
    opt['use_rot'] = True

    opt['use_shuffle'] = True
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 16
    opt['scale'] = 4

    opt['dataset_enlarge_ratio'] = 1

    mmcv.mkdir_or_exist('tmp')

    dataset = create_dataset(opt)
    data_loader = create_dataloader(
        dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(i)

        lq = data['lq']
        gt = data['gt']
        key = data['key']
        print(key)
        for j in range(opt['num_frame']):
            torchvision.utils.save_image(
                lq[:, j, :, :, :],
                f'tmp/lq_{i:03d}_frame{j}.png',
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
