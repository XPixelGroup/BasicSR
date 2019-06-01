import os.path as osp
import random
import pickle
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util


class LRHRDataset(data.Dataset):
    '''
    Read LR and GT image pairs.
    If only GT image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LR, self.paths_GT = None, None
        self.LR_env, self.GT_env = None, None  # environment for lmdb
        if self.data_type == 'img':
            self.paths_GT = sorted(util._get_paths_from_images(opt['dataroot_GT']))
            self.paths_LR = sorted(util._get_paths_from_images(opt['dataroot_LR']))
        elif self.data_type == 'lmdb':
            keys = pickle.load(open(osp.join(opt['dataroot_GT'], '_keys_cache.p'), 'rb'))
            self.paths_GT = sorted([key for key in keys if not key.endswith('.meta')])
            keys = pickle.load(open(osp.join(opt['dataroot_LR'], '_keys_cache.p'), 'rb'))
            self.paths_LR = sorted([key for key in keys if not key.endswith('.meta')])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LR and self.paths_GT:
            assert len(self.paths_LR) == len(self.paths_GT), \
                'GT and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_GT))
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False, meminit=False)
        self.LR_env = lmdb.open(
            self.opt['dataroot_LR'], readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LR_env is None):
                self._init_lmdb()
        GT_path, LR_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT = util.read_img(self.GT_env, GT_path)
        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_GT = util.modcrop(img_GT, scale)
        # change color space if necessary
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get LR image
        if self.paths_LR:
            LR_path = self.paths_LR[index]
            img_LR = util.read_img(self.LR_env, LR_path)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LR = util.imresize_np(img_GT, 1 / scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(
                    np.copy(img_GT), (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LR = util.imresize_np(img_GT, 1 / scale, True)
                if img_LR.ndim == 2:
                    img_LR = np.expand_dims(img_LR, axis=2)

            H, W, C = img_LR.shape
            LR_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LR, img_GT = util.augment([img_LR, img_GT], self.opt['use_flip'], \
                self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'],
                                          [img_LR])[0]  # TODO during val no definition

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = GT_path
        return {'LR': img_LR, 'GT': img_GT, 'LR_path': LR_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
