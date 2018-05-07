import os.path
import random
import time
import cv2
import numpy as np

import torch
import torch.utils.data as data
from data.util import get_image_paths


class LRHRRefDataset(data.Dataset):

    def name(self):
        return 'LRHRRefDataset'

    def __init__(self, opt):
        super(LRHRRefDataset, self).__init__()
        self.opt = opt
        self.paths_LR = []
        self.paths_HR = []
        self.paths_ref = []

        if opt['data_type'] == 'lmdb': # only used in train phase
            import lmdb
            if opt['dataroot_LR'] is not None:
                self.LR_env = lmdb.open(opt['dataroot_LR'], readonly=True, \
                        lock=False, readahead=False, meminit=False)
                with self.LR_env.begin(write=False) as txn:
                    keys = [key.decode('ascii') for key, _ in txn.cursor()]
                self.paths_LR = sorted([key for key in keys if not key.endswith('.meta')])
            if opt['dataroot_HR'] is not None:
                self.HR_env = lmdb.open(opt['dataroot_HR'], readonly=True, \
                        lock=False, readahead=False, meminit=False)
                with self.HR_env.begin(write=False) as txn:
                    keys = [key.decode('ascii') for key, _ in txn.cursor()]
                self.paths_HR = sorted([key for key in keys if not key.endswith('.meta')])
            if self.paths_LR and self.paths_HR:
                assert len(self.paths_LR) == len(self.paths_HR), \
                    'HR and LR lmdb datasets have different number of images.'
            # TODO lmdb does not support ref image now
        else: # read image from files
            if opt['phase'] == 'train' and opt['subset_file'] is not None:
                # get HR image paths from list
                with open(opt['subset_file']) as f:
                    self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) \
                            for line in f])
                if opt['dataroot_LR'] is not None:
                    raise NotImplementedError('Now subset only support generating LR on-the-fly.')
            else:
                if opt['dataroot_LR'] is not None:
                    self.paths_LR = sorted(get_image_paths(opt['dataroot_LR']))
                if opt['dataroot_HR'] is not None:
                    self.paths_HR = sorted(get_image_paths(opt['dataroot_HR']))
                assert self.paths_LR or self.paths_HR, 'Both LR and HR paths are empty.'
                if self.paths_LR and self.paths_HR:
                    assert len(self.paths_LR) == len(self.paths_HR), \
                        'HR and LR datasets have different number of images.'
            # ref images
            if opt['dataroot_ref']:
                self.paths_ref = sorted(get_image_paths(opt['dataroot_ref']))

        # self.aug_scale_list = [] # may need to incorporate into the option, now it is useless

    def __getitem__(self, index):
        HR_path, LR_path, ref_path = None, None, None
        scale = self.opt['scale']

        if self.paths_HR:
            # get HR image
            if self.opt['data_type'] == 'img':
                HR_path = self.paths_HR[index]
                img_HR = cv2.imread(HR_path, cv2.IMREAD_UNCHANGED)
            else:  # lmdb
                with self.HR_env.begin(write=False) as txn:
                    HR_path = self.paths_HR[index]
                    buf = txn.get(HR_path.encode('ascii'))
                    buf_meta = txn.get((HR_path + '.meta').encode('ascii')).decode('ascii')
                img_HR_flat = np.frombuffer(buf, dtype=np.uint8)
                H, W, C = [int(s) for s in buf_meta.split(',')]
                img_HR = img_HR_flat.reshape(H, W, C)
            img_HR = img_HR * 1.0 / 255  # numpy.ndarray(float64), [0,1], HWC, BGR
            if img_HR.ndim == 2:  # gray image, add one dimension
                img_HR = np.expand_dims(img_HR, axis=2)

            # get LR image
            if self.paths_LR:
                if self.opt['data_type'] == 'img':
                    LR_path = self.paths_LR[index]
                    img_LR = cv2.imread(LR_path, cv2.IMREAD_UNCHANGED)
                else:  # lmdb
                    with self.LR_env.begin(write=False) as txn:
                        LR_path = self.paths_LR[index]
                        buf = txn.get(LR_path.encode('ascii'))
                        buf_meta = txn.get((LR_path + '.meta').encode('ascii')).decode('ascii')
                    img_LR_flat = np.frombuffer(buf, dtype=np.uint8)
                    H, W, C = [int(s) for s in buf_meta.split(',')]
                    img_LR = img_LR_flat.reshape(H, W, C)
                img_LR = img_LR * 1.0 / 255
            # down-sampling on-the-fly
            else:
                H, W, _ = img_HR.shape
                # using INTER_LINEAR now
                img_LR = cv2.resize(img_HR, (W//scale, H//scale), interpolation=cv2.INTER_LINEAR)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        # only read LR image in test phase
        else:
            LR_path = self.paths_LR[index]
            img_LR = cv2.imread(LR_path, cv2.IMREAD_UNCHANGED)
            img_LR = img_LR * 1.0 / 255
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        H, W, C = img_LR.shape
        if self.opt['phase'] == 'train':
            HR_size = self.opt['HR_size']
            LR_size = HR_size // scale

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

            # read ref images
            if self.opt['dataroot_ref']:
                ref_idx = random.randint(0, len(self.paths_ref) - 1)
                ref_path = self.paths_ref[ref_idx]
                img_ref = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
                img_ref = img_ref * 1.0 / 255
                if img_ref.ndim == 2:
                    img_ref = np.expand_dims(img_ref, axis=2)
                ref_size = LR_size if self.opt['reverse'] else HR_size
                rnd_h = random.randint(0, max(0, img_ref.shape[0] - ref_size))
                rnd_w = random.randint(0, max(0, img_ref.shape[1] - ref_size))
                img_ref = img_ref[rnd_h:rnd_h + ref_size, rnd_w:rnd_w + ref_size, :]
                img_ref = np.ascontiguousarray(img_ref)
                img_ref = _augment(img_ref)
                img_ref = torch.from_numpy(np.transpose(img_ref[:, :, [2, 1, 0]], (2, 0, 1))).float()

        # numpy to tensor, HWC to CHW, BGR to RGB
        if HR_path:
            if C == 3:
                img_HR = torch.from_numpy(np.transpose(img_HR[:, :, [2,1,0]], (2, 0, 1))).float()
            else:
                img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        # must have LR image
        if C == 3:
            img_LR = torch.from_numpy(np.transpose(img_LR[:, :, [2,1,0]], (2, 0, 1))).float()
        else:
            img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        if not HR_path:  # read only LR image in test phase
            return {'LR': img_LR, 'LR_path': LR_path}
        elif LR_path is None:
            LR_path = HR_path # tricky
        if self.opt['reverse']:
            img_LR, img_HR = img_HR, img_LR
            LR_path, HR_path = HR_path, LR_path
        return_dict = {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}
        if ref_path:
            return_dict['ref'] = img_ref
        return return_dict

    def __len__(self):
        if self.paths_LR:
            return len(self.paths_LR)
        else:
            return len(self.paths_HR)
