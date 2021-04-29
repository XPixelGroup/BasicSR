import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import cv2
from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import (
    FileClient, imfrombytes, img2tensor, scandir,
    Options, dict2object, object2dict
)
import glob
import os
import numpy as np
from tqdm import tqdm
import imgaug.augmenters as ia
from imgaug.augmenters.meta import Augmenter  # baseclass

# TODO: a 16x class with an `augment_image` method


class Mosaic_16x:
    def augment_image(self, x):
        h, w = x.shape[:2]
        x = x.astype('float')  # avoid overflow for uint8
        for i in range((h + 15) // 16):
            for j in range((w + 15) // 16):
                mean = x[i * 16:(i + 1) * 16, j * 16:(j + 1)
                         * 16].mean(axis=(0, 1))
                x[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = mean

        return x.astype('uint8')


class Degradation_Simulator:
    """
    [Lotayou] 20210424: Generating training/testing data pairs on the fly
    The degradation script is aligned with HiFaceGAN paper settings.

    Args:
        opt(str | op): Config for degradation script, with degradation type and parameters
        Custom degradation is possible by passing an inherited class from ia.augmentors
    """

    def __init__(self,):
        DEFAULT_DEG_TEMPLATES = {
            'sr4x': ia.Sequential([
                # It's almost like a 4x bicubic downsampling
                ia.Resize((0.2500, 0.2501)),
                ia.Resize({"height": 512, "width": 512}),
            ]),

            'sr4x8x': ia.Sequential([
                ia.Resize((0.125, 0.25)),
                ia.Resize({"height": 512, "width": 512}),
            ]),

            'denoise': ia.OneOf([
                ia.AdditiveGaussianNoise(scale=(20, 40), per_channel=True),
                ia.AdditiveLaplaceNoise(scale=(20, 40), per_channel=True),
                ia.AdditivePoissonNoise(lam=(15, 30), per_channel=True),
            ]),

            'deblur': ia.OneOf([
                ia.MotionBlur(k=(10, 20)),
                ia.GaussianBlur((3.0, 8.0)),
            ]),

            'jpeg': ia.JpegCompression(compression=(50, 85)),
            '16x': Mosaic_16x(),
            # This ain't gonna pass
            'face_renov': self._get_rand()
        }

    def _get_rand(self,):
        return ia.Sequential([
            self.DEFAULT_DEG_TEMPLATES['deblur'],
            self.DEFAULT_DEG_TEMPLATES['denoise'],
            self.DEFAULT_DEG_TEMPLATES['jpeg'],
            self.DEFAULT_DEG_TEMPLATES['sr4x8x'],
        ], random_order=True)

    def create_training_dataset(self, deg, src_folder, dest_folder=None):
        '''
            Create a degradation simulator and
            apply it to GT images on the fly
        '''
        if isinstance(deg, str):
            assert deg in self.DEFAULT_DEG_TEMPLATES, \
                'Degration type %s not recognized: (%s)' % \
                (deg, '|'.join(list(self.DEFAULT_DEG_TEMPLATES.keys())))
            deg = self.DEFAULT_DEG_TEMPLATES[deg]
        else:
            assert isinstance(deg, Augmenter), \
                'Deg must be either str|Augmenter, got %s' % type(deg)

        if not dest_folder:
            suffix = deg if isinstance(deg, str) else 'custom'
            dest_folder = '_'.join([src_folder, suffix])

        names = os.listdir(src_folder)
        for name in tqdm(names):
            gt = cv2.imread(os.path.join(src_folder, name))
            lq = deg.augment_image(gt)
            pack = np.concatenate([lq, gt], axis=0)
            cv2.imwrite(os.path.join(dest_folder, name), pack)

        print('Dataset prepared.')


class HiFaceGANDataset(data.Dataset):
    """
    [Lotayou] 20210424: Customized dataset for training and evaluation of HiFaceGAN.

    Args:
        opt (dict or Option): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.

    Note the official HiFaceGAN project (https://github.com/Lotayou/Face-Renovation)
    assumes paired input arranged along the height dimension ([LQ; HQ])
    Say for a 512*512 input, an aligned pair has size 1024*512*3

    Also, HiFaceGAN manages options with object classes instead of yaml files
    so we call args as opt.XXX instead of opt['XXX']

    Future works:
    [ ] lmdb support
    [ ] yml support
    [-] allow reading from two separate folders (LQ/HQ) with inconsistent sizes

    """

    def __init__(self, opt):
        super(HiFaceGANDataset, self).__init__()
        # yaml compatibility
        if isinstance(opt, dict):
            opt = dict2object(opt)

        print(opt)

        assert opt.phase == 'test' or opt.with_gt, 'Training requires opt.with_gt=True'
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.gt_folder = opt.dataroot
        # Set opt.io_backend as a dict then
        self.io_backend_opt = opt.io_backend

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', "
                                 f'but received {self.gt_folder}')
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = sorted([
                l for l in glob.glob(os.path.join(opt.dataroot, '*'))
                if l.lower().endswith(('jpg', 'png', 'jpeg'))
            ])

        # dataloader does not support collate Nonetype
        self.dummy = torch.tensor(0)

    def __getitem__(self, index):

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'),
                **self.io_backend_opt
            )

        path = self.paths[index]
        img_bytes = self.file_client.get(path)
        image = imfrombytes(img_bytes, float32=True, bgr2rgb=True)

        h, w = image.shape[:2]
        s = min(h, w)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).to(torch.float32)
        lq = image[:, :s]  # left or top, same indices
        gt = image[:, s:] if self.opt.with_gt else self.dummy

        input_dict = {
            'label': lq,
            'image': gt,
            'path': path,
        }
        return input_dict

    def __len__(self,):
        return len(self.paths)

# Unit test
# if __name__ == '__main__':
