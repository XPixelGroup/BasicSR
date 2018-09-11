import time
import math
import numpy as np
import torch
import torchvision.utils
from data import create_dataloader, create_dataset
from utils import util

opt = {}

opt['name'] = 'DIV2K800'
opt['dataroot_HR'] = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb'
opt['dataroot_LR'] = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub_bicLRx4.lmdb'

# opt['dataroot_HR'] = '/mnt/SSD/xtwang/BasicSR_datasets/OST/train/img'
# opt['dataroot_HR_bg'] = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub'
# opt['dataroot_LR'] = None

opt['subset_file'] = None
opt['mode'] = 'LRHR'  # 'LRHR' | 'LRHRseg_bg'
opt['phase'] = 'train'  # 'train' | 'val'
opt['use_shuffle'] = True
opt['n_workers'] = 8
opt['batch_size'] = 16
opt['HR_size'] = 96
opt['scale'] = 4
opt['use_flip'] = True
opt['use_rot'] = True
opt['color'] = 'RGB'

opt['data_type'] = 'lmdb'  # img lmdb

# for segmentation
# look_up table # RGB
lookup_table = torch.from_numpy(
    np.array([
        [153, 153, 153],  # 0, background
        [0, 255, 255],  # 1, sky
        [109, 158, 235],  # 2, water
        [183, 225, 205],  # 3, grass
        [153, 0, 255],  # 4, mountain
        [17, 85, 204],  # 5, building
        [106, 168, 79],  # 6, plant
        [224, 102, 102],  # 7, animal
        [255, 255, 255],  # 8/255, void
    ])).float()
lookup_table /= 255


def render(seg):
    _, argmax = torch.max(seg, 0)
    argmax = argmax.squeeze().byte()
    # color img
    im_h, im_w = argmax.size()
    color = torch.FloatTensor(3, im_h, im_w).fill_(0)  # black
    for k in range(8):
        mask = torch.eq(argmax, k)
        color.select(0, 0).masked_fill_(mask, lookup_table[k][0])  # R
        color.select(0, 1).masked_fill_(mask, lookup_table[k][1])  # G
        color.select(0, 2).masked_fill_(mask, lookup_table[k][2])  # B
    # void
    mask = torch.eq(argmax, 255)
    color.select(0, 0).masked_fill_(mask, lookup_table[8][0])  # R
    color.select(0, 1).masked_fill_(mask, lookup_table[8][1])  # G
    color.select(0, 2).masked_fill_(mask, lookup_table[8][2])  # B
    return color


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

    if opt['mode'] == 'LRHRseg_bg':
        seg = data['seg']
        seg_color_list = []
        for j in range(seg.size(0)):
            _seg = seg[j, :, :, :]
            seg_color_list.append(render(_seg).unsqueeze_(0))

        seg_color_batch = torch.cat(seg_color_list, 0)
        torchvision.utils.save_image(
            seg_color_batch, 'tmp/seg_{:03d}.png'.format(i), nrow=nrow, padding=2, normalize=False)
