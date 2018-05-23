import os.path
import cv2
import numpy as np
import torch
import torch.utils.data as data
import data.util as util


class LRDataset(data.Dataset):
    '''
    Read LR only for testing.
    '''

    def name(self):
        return 'LRDataset'

    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = []

        # read lmdb files
        if opt['data_type'] == 'lmdb':
            self.LR_env, self.paths_LR = util.get_paths_from_lmdb(opt['dataroot_LR'])
        else:  # read image files
            self.paths_LR = sorted(util.get_image_paths(opt['dataroot_LR']))
        assert self.paths_LR, 'Error: LR paths are empty.'  # must have LR paths.

    def __getitem__(self, index):
        LR_path = None

        # get LR image
        LR_path = self.paths_LR[index]
        if self.opt['data_type'] == 'img':
            img_LR = cv2.imread(LR_path, cv2.IMREAD_UNCHANGED)
        else:  # lmdb
            img_LR = util.read_lmdb_img(self.LR_env, LR_path)
        img_LR = img_LR * 1.0 / 255
        H, W, C = img_LR.shape

        # channel conversion
        if C == 3 and self.opt['color'] == 'gray':  # RGB to gray
            img_LR = np.dot(img_LR[..., :3], [0.2989, 0.587, 0.114])
            img_LR = np.expand_dims(img_LR, axis=2)
        elif C == 3 and self.opt['color'] == 'y':  # RGB to y
            img_LR = np.dot(img_LR[..., :3],
                            [65.481 / 255, 128.553 / 255, 24.966 / 255]) + 16.0 / 255
            img_LR = np.expand_dims(img_LR, axis=2)
            img_LR = np.repeat(img_LR, 3, axis=2)

        # numpy to tensor, HWC to CHW, BGR to RGB
        if img_LR.shape[2] == 3:
            img_LR = torch.from_numpy(np.transpose(img_LR[:, :, [2, 1, 0]], (2, 0, 1))).float()
        else:
            img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        return {'LR': img_LR, 'LR_path': LR_path}

    def __len__(self):
        return len(self.paths_LR)
