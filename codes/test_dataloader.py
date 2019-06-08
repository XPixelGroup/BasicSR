import math
import torchvision.utils
from data import create_dataloader, create_dataset
from utils import util

opt = {}

opt['name'] = 'DIV2K800'
opt['dataroot_GT'] = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb'
opt['dataroot_LQ'] = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub_bicLRx4.lmdb'
opt['mode'] = 'LQGT'
opt['phase'] = 'train'  # 'train' | 'val'
opt['use_shuffle'] = True
opt['n_workers'] = 8
opt['batch_size'] = 16
opt['GT_size'] = 128
opt['scale'] = 4
opt['use_flip'] = True
opt['use_rot'] = True
opt['color'] = 'RGB'
opt['data_type'] = 'lmdb'  # img | lmdb
opt['dist'] = False
opt['gpu_ids'] = [0]

util.mkdir('tmp')
train_set = create_dataset(opt)
train_loader = create_dataloader(train_set, opt, opt, None)
nrow = int(math.sqrt(opt['batch_size']))
if opt['phase'] == 'train':
    padding = 2
else:
    padding = 0

for i, data in enumerate(train_loader):
    if i > 5:
        break
    print(i)
    LQ = data['LQ']
    GT = data['GT']
    torchvision.utils.save_image(LQ, 'tmp/LQ_{:03d}.png'.format(i), nrow=nrow, padding=padding,
                                 normalize=False)
    torchvision.utils.save_image(GT, 'tmp/GT_{:03d}.png'.format(i), nrow=nrow, padding=padding,
                                 normalize=False)
