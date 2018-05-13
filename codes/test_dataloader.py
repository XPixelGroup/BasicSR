import time
import math
import torchvision.utils
from data import create_dataloader, create_dataset


opt = {}

# # subset and HR path only
# opt['name'] = 'ImageNet'
# opt['dataroot_HR'] = '/mnt/SSD/xtwang/ImageNet_train'
# opt['dataroot_LR'] = None
# opt['subset_file'] = '/mnt/SSD/xtwang/BasicSR_datasets/ImageNet_list.txt'

opt['name'] = 'DIV2K'
opt['dataroot_HR'] = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb'
opt['dataroot_LR'] = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub_bicLRx4.lmdb'
opt['subset_file'] = None

opt['dataroot_ref'] = None
opt['reverse'] = False

opt['data_type'] = 'lmdb'
opt['mode'] = 'LRHRref'
opt['phase'] = 'train'
opt['use_shuffle'] = True
opt['n_workers'] = 8
opt['batch_size'] = 16
opt['HR_size'] = 192
opt['scale'] = 4
opt['use_flip'] = True
opt['use_rot'] = True

train_set = create_dataset(opt)
train_loader = create_dataloader(train_set, opt)
nrow = int(math.sqrt(opt['batch_size']))
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
    if 'ref' in data:
        ref = data['ref']
        torchvision.utils.save_image(ref, 'ref_{:03d}.png'.format(i), nrow=nrow, padding=2, \
            normalize=False)
    torchvision.utils.save_image(LR, 'LR_{:03d}.png'.format(i), nrow=nrow, padding=2, normalize=False)
    torchvision.utils.save_image(HR, 'HR_{:03d}.png'.format(i), nrow=nrow, padding=2, normalize=False)
