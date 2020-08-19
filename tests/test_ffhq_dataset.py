import math
import mmcv
import torch
import torchvision.utils

from basicsr.data import create_dataloader, create_dataset


def main():
    """Test FFHQ dataset."""
    opt = {}
    opt['dist'] = False
    opt['gpu_ids'] = [0]
    opt['phase'] = 'train'

    opt['name'] = 'FFHQ'
    opt['type'] = 'FFHQDataset'

    opt['dataroot_gt'] = 'datasets/ffhq/ffhq_256.lmdb'
    opt['io_backend'] = dict(type='lmdb')

    opt['use_hflip'] = True
    opt['mean'] = [0.5, 0.5, 0.5]
    opt['std'] = [0.5, 0.5, 0.5]

    opt['use_shuffle'] = True
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 4

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

        gt = data['gt']
        print(torch.min(gt), torch.max(gt))
        gt_path = data['gt_path']
        print(gt_path)
        torchvision.utils.save_image(
            gt,
            f'tmp/gt_{i:03d}.png',
            nrow=nrow,
            padding=padding,
            normalize=True,
            range=(-1, 1))


if __name__ == '__main__':
    main()
