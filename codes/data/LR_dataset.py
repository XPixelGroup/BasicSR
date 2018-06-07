import os.path
import cv2
import numpy as np
import torch
import torch.utils.data as data
import data.util as util


class LRDataset(data.Dataset):
    '''
    Read LR images only for testing.
    '''

    def name(self):
        return 'LRDataset'

    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = []

        # read image list from lmdb or image files
        if opt['data_type'] == 'lmdb':
            self.LR_env, self.paths_LR = util.get_paths_from_lmdb(opt['dataroot_LR'])
        else:
            self.paths_LR = sorted(util.get_image_paths(opt['dataroot_LR']))
        assert self.paths_LR, 'Error: LR paths are empty.'

    def __getitem__(self, index):
        LR_path = None

        # get LR image
        LR_path = self.paths_LR[index]
        if self.opt['data_type'] == 'img':
            img_LR = cv2.imread(LR_path, cv2.IMREAD_UNCHANGED)
        else:
            img_LR = util.read_lmdb_img(self.LR_env, LR_path)
        img_LR = img_LR.astype(np.float32) / 255.
        if img_LR.ndim == 2:
            img_LR = np.expand_dims(img_LR, axis=2)
        H, W, C = img_LR.shape

        # channel conversion
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0]

        # HWC to CHW, BGR to RGB, numpy to tensor
        if img_LR.shape[2] == 3:
            img_LR = cv2.cvtColor(img_LR, cv2.COLOR_BGR2RGB)
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        return {'LR': img_LR, 'LR_path': LR_path}

    def __len__(self):
        return len(self.paths_LR)
