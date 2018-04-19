import os.path
import random
import time
import cv2
import numpy as np

import torch
import torch.utils.data as data
from data.util import get_image_paths

class LRHRPairDataset(data.Dataset):
    def __init__(self, opt):
        super(LRHRPairDataset, self).__init__()
        self.opt = opt
        self.paths_LR = []
        self.paths_HR = []

        if opt['phase'] == 'train' and opt['subset_file'] is not None:
            # get HR image paths from list
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) \
                        for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('subset only support generating LR now.')
        else:
            if opt['dataroot_LR'] is not None:
                self.paths_LR = sorted(get_image_paths(opt['dataroot_LR']))
            if opt['dataroot_HR'] is not None:
                self.paths_HR = sorted(get_image_paths(opt['dataroot_HR']))
            assert self.paths_LR or self.paths_HR, 'both LR and HR paths are empty.'
            if self.paths_LR and self.paths_HR:
                assert len(self.paths_LR) == len(self.paths_HR), 'HR and LR datasets are not the same.'

        if opt['data_type'] == 'bin':
            # only in train, not val and test
            self.HR_bin_list = []
            self.LR_bin_list = []
            # TODO

        self.aug_scale_list = [] # TODO may need to incorporate into the option, now it is useless

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt['scale']
        if self.paths_HR:
            # get HR image
            if self.opt['data_type'] == 'img':
                HR_path = self.paths_HR[index]
                img_HR = cv2.imread(HR_path, cv2.IMREAD_UNCHANGED)
            else: # bin
                img_HR = self.HR_bin_list[index]
            img_HR = img_HR * 1.0 / 255 # numpy.ndarray(float64), [0,1], HWC, BGR
            if img_HR.ndim == 2: # gray image, add one dimension
                img_HR = np.expand_dims(img_HR, axis=2)

            # get LR image
            if self.paths_LR:
                if self.opt['data_type'] == 'img':
                    LR_path = self.paths_LR[index]
                    img_LR = cv2.imread(LR_path, cv2.IMREAD_UNCHANGED)
                else: # bin
                    img_LR = self.LR_bin_list[index]
                img_LR = img_LR * 1.0 / 255  # numpy.ndarray(float64), [0,1], HWC, BGR
                if img_LR.ndim == 2:  # gray image, add one dimension
                    img_LR = np.expand_dims(img_LR, axis=2)
            else: # down-sampling on-the-fly
                h_HR, w_HR, _ = img_HR.shape
                # using bilinear now
                img_LR = cv2.resize(img_HR, (w_HR//scale, h_HR//scale), interpolation=cv2.INTER_LINEAR)
        else: # only read LR image in test phase
            LR_path = self.paths_LR[index]
            img_LR = cv2.imread(LR_path, cv2.IMREAD_UNCHANGED)
            img_LR = img_LR * 1.0 / 255
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        H, W, C = img_LR.shape

        if self.opt['phase'] == 'train':
            HR_size = self.opt['HR_size']
            LR_size = HR_size // scale

            # randomly scale
            # if self.aug_scale_list != []:
            #     rnd_scale = random.choice(self.scale_list)
            #     H, W = int(rnd_scale*H), int(rnd_scale*W)
            #     if H < LR_size:
            #         H = LR_size
            #     if W < LR_size:
            #         W = LR_size
            #     img_LR = cv2.resize(img_LR, (W,H), interpolation=cv2.INTER_LINEAR)
            #     img_HR = cv2.resize(img_HR, (W*self.opt.scale,H*self.opt.scale), \
            #           interpolation=cv2.INTER_LINEAR)

            # randomly crop
            rnd_h = random.randint(0, max(0, H-LR_size))
            rnd_w = random.randint(0, max(0, W-LR_size))
            img_LR = img_LR[rnd_h:rnd_h+LR_size, rnd_w:rnd_w+LR_size, :]
            rnd_h_HR, rnd_w_HR = rnd_h*scale, rnd_w*scale
            img_HR = img_HR[rnd_h_HR:rnd_h_HR+HR_size, rnd_w_HR:rnd_w_HR+HR_size, :]
            img_LR = np.ascontiguousarray(img_LR)
            img_HR = np.ascontiguousarray(img_HR)

            # augmentation - flip, rotation
            hflip = self.opt['use_flip'] and random.random() > 0.5
            vflip = self.opt['use_rot'] and random.random() > 0.5
            rot90 = self.opt['use_rot'] and random.random() > 0.5
            def _augment(img):
                if hflip: img = img[:, ::-1, :]
                if vflip: img = img[::-1, :, :]
                if rot90: img = img.transpose(1, 0, 2)
                return img
            img_HR = _augment(img_HR)
            img_LR = _augment(img_LR)

        # numpy to tensor, HWC to CHW, BGR to RGB
        if self.paths_HR:
            img_HR = torch.from_numpy(np.transpose(img_HR[:, :, [2,1,0]], (2, 0, 1))).float()
        img_LR = torch.from_numpy(np.transpose(img_LR[:, :, [2,1,0]], (2, 0, 1))).float()

        if not self.paths_HR: # read only LR image in test phase
            return {'LR': img_LR, 'LR_path': LR_path}
        elif  LR_path is None:
            LR_path = HR_path
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        if self.paths_LR:
            return len(self.paths_LR)
        else:
            return len(self.paths_HR)

    def name(self):
        return 'LRHRPairDataset'
