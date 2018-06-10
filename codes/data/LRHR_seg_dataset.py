import os.path
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
import data.util as util


class LRHRSegDataset(data.Dataset):
    '''
    Read HR, seg; generate LR, category
    for SFT-GAN
    '''

    def name(self):
        return 'LRHRSegDataset'

    def __init__(self, opt):
        super(LRHRSegDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None

        # read image list from lmdb or image files
        self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])
        self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])

        assert self.paths_HR, 'Error: HR paths are empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR))

        # randomly scale list
        self.random_scale_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5]


    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']

        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(self.HR_env, HR_path)
        # modcrop in validation phase
        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, 8)

        # get segmentation probability map
        seg = torch.load(HR_path.replace('/img/', '/bicseg/').replace('.png', '.pth'))
        seg = np.transpose(seg.numpy(), (1, 2, 0))

        # get LR image
        if self.paths_LR:
            LR_path = self.paths_LR[index]
            img_LR = util.read_img(self.LR_env, LR_path)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = seg.shape
                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt
                H_s = _mod(H_s, random_scale, scale, HR_size)
                W_s = _mod(W_s, random_scale, scale, HR_size)
                img_HR = cv2.resize(np.copy(img_HR), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                seg = cv2.resize(np.copy(seg), (W_s, H_s), interpolation=cv2.INTER_NEAREST)

            H, W, _ = img_HR.shape
            # using matlab imresize
            img_LR = util.imresize_np(img_HR, 1 / scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        H, W, C = img_LR.shape
        if self.opt['phase'] == 'train':
            LR_size = HR_size // scale
            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]
            seg = seg[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]

            # augmentation - flip, rotate
            img_LR, img_HR, seg = util.augment([img_LR, img_HR, seg], self.opt['use_flip'],
                                          self.opt['use_rot'])

            # category
            if 'building' in HR_path:
                category = 0
            elif 'plant' in HR_path:
                category = 1
            elif 'mountain' in HR_path:
                category = 2
            elif 'water' in HR_path:
                category = 3
            elif 'sky' in HR_path:
                category = 4
            elif 'grass' in HR_path:
                category = 5
            elif 'animal' in HR_path:
                category = 6
            else:
                category = 7 # background
        else:
            category = -1 # during val, useless

        # HWC to CHW, BGR to RGB, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        seg = torch.from_numpy(np.ascontiguousarray(np.transpose(seg, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = HR_path
        return {'LR': img_LR, 'HR': img_HR, 'seg': seg, 'category':category,
            'LR_path': LR_path, 'HR_path': HR_path}


    def __len__(self):
        return len(self.paths_HR)
