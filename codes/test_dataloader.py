import time
import math
import torchvision.utils
from data import create_dataloader, create_dataset
from utils import util

opt = {}

opt['name'] = 'test'
opt['dataroot_HR'] = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb'
opt['dataroot_LR'] = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub_bicLRx4.lmdb'
opt['subset_file'] = None
opt['mode'] = 'LRHR'
opt['phase'] = 'train'
opt['use_shuffle'] = True
opt['n_workers'] = 8
opt['batch_size'] = 16
opt['HR_size'] = 192
opt['scale'] = 4
opt['use_flip'] = True
opt['use_rot'] = True
opt['color'] = 'RGB'

opt['data_type'] = 'lmdb'

util.mkdir('tmp')
train_set = create_dataset(opt)
train_loader = create_dataloader(train_set, opt)
nrow = int(math.sqrt(opt['batch_size']))
if opt['phase'] == 'train':
    padding = 2
else:
    padding = 0

for i, data in enumerate(train_loader):
    # test dataloader time
    # if i == 1:
    #     start_time = time.time()
    # if i == 500:
    #     print(time.time() - start_time)
    #     break
    if i > 5:
        break
    print(i)
    LR = data['LR']
    HR = data['HR']
    torchvision.utils.save_image(
        LR, 'tmp/LR_{:03d}.png'.format(i), nrow=nrow, padding=padding, normalize=False)
    torchvision.utils.save_image(
        HR, 'tmp/HR_{:03d}.png'.format(i), nrow=nrow, padding=padding, normalize=False)
