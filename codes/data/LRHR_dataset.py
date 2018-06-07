import os.path
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
import data.util as util


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR image pair.
    If only HR image is provided, generate LR image on-the-fly.
    The pair relation is ensured by 'sorted' function, so please check the name convention.
    '''

    def name(self):
        return 'LRHRDataset'

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = []
        self.paths_HR = []

        # read image list from lmdb or image files
        if opt['data_type'] == 'lmdb':
            if opt['dataroot_LR'] is not None:
                self.LR_env, self.paths_LR = util.get_paths_from_lmdb(opt['dataroot_LR'])
            if opt['dataroot_HR'] is not None:
                self.HR_env, self.paths_HR = util.get_paths_from_lmdb(opt['dataroot_HR'])
        else:
            if opt['phase'] == 'train' and opt['subset_file'] is not None:  # from subset list
                with open(opt['subset_file']) as f:
                    self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) \
                            for line in f])
                if opt['dataroot_LR'] is not None:
                    raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
            else:
                if opt['dataroot_LR'] is not None:
                    self.paths_LR = sorted(util.get_image_paths(opt['dataroot_LR']))
                if opt['dataroot_HR'] is not None:
                    self.paths_HR = sorted(util.get_image_paths(opt['dataroot_HR']))

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

        # get HR image
        HR_path = self.paths_HR[index]
        if self.opt['data_type'] == 'img':
            img_HR = cv2.imread(HR_path, cv2.IMREAD_UNCHANGED)
        else:
            img_HR = util.read_lmdb_img(self.HR_env, HR_path)
        img_HR = img_HR.astype(np.float32) / 255.
        if img_HR.ndim == 2:
            img_HR = np.expand_dims(img_HR, axis=2)

        # get LR image
        if self.paths_LR:
            LR_path = self.paths_LR[index]
            if self.opt['data_type'] == 'img':
                img_LR = cv2.imread(LR_path, cv2.IMREAD_UNCHANGED)
            else:
                img_LR = util.read_lmdb_img(self.LR_env, LR_path)
            img_LR = img_LR.astype(np.float32) / 255.
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                HR_size = self.opt['HR_size']
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_HR.shape
                H_s, W_s = int(H_s * random_scale), int(W_s * random_scale)
                if H_s < HR_size:
                    H_s = HR_size
                if W_s < HR_size:
                    W_s = HR_size
                img_HR = cv2.resize(img_HR, (W_s, H_s), interpolation=cv2.INTER_LINEAR)

            H, W, _ = img_HR.shape
            # using matlab imresize
            img_LR = util.imresize_np(img_HR, 1 / scale, True)
            # img_LR = cv2.resize(img_HR, (W // scale, H // scale), interpolation=cv2.INTER_LINEAR)
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

            # augmentation - flip, rotate
            img_LR, img_HR = util.augment([img_LR, img_HR], self.opt['use_flip'], self.opt['use_rot'])

        # channel conversion
        if self.opt['color']:
            img_LR, img_HR = util.channel_convert(C, self.opt['color'], [img_LR, img_HR])

        # HWC to CHW, BGR to RGB, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = cv2.cvtColor(img_HR, cv2.COLOR_BGR2RGB)
            img_LR = cv2.cvtColor(img_LR, cv2.COLOR_BGR2RGB)
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = 'on-the-fly'
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.paths_HR)
