import os.path as osp

import mmcv
import numpy as np
import torch.utils.data as data

from basicsr.data.transforms import totensor
from basicsr.utils import FileClient


class SingleImageDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'ann_file': Use annotation file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            ann_file (str): Path for annotation file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.lq_folder = opt['dataroot_lq']
        if 'ann_file' in self.opt:
            with open(self.opt['ann_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.lq_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = [
                osp.join(self.lq_folder, v)
                for v in mmcv.scandir(self.lq_folder)
            ]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path)
        img_lq = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = totensor(img_lq, bgr2rgb=True, float32=True)

        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)
