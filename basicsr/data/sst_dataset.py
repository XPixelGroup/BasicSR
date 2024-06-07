import torch
import os.path as osp
import numpy as np
from tqdm import tqdm
import netCDF4 as nc

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from datetime import datetime
import matplotlib.pyplot as plt


def is_leap_year(year: int):
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0

def cesm_dt2idx(dt: datetime):
    start_date = datetime(1850, 1, 1)
    idx = (dt.year - start_date.year) * 365
    idx += (dt - datetime(dt.year, 1, 1)).days
    if is_leap_year(dt.year) and dt.month > 2:
        idx -= 1
    return idx
    
    
@DATASET_REGISTRY.register()
class sstDataset(data.Dataset):
    """CESM and ERA5 SST state for meterological field reconstruction. Land areas are masked out.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    The CESM data is arranged in a single file, each CESM file contains data from 18500101 to 20051231, a typical file
    name is: b.e11.BLMTRC5CN.f19_g16.002.cam.h1.TREFHT.18500101-20051231.nc.
    
    The ERA5 data is arranged monthly, each ERA5 file contains 1 month of data, a typical file name is:
    era5_t2m_1948_10.npy, and its shape is (31, 721, 1440), where 31 is the number of days in October 1948, 721 is the
    latitude grids and 1440 is the longitude grids.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        start_date (str): Start date for the dataset.
        end_date (str): End date for the dataset.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(sstDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        assert self.mean is not None and self.std is not None, 'mean and std must be provided in the config file'

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk', 'Only support disk io backend for t2m dataset'
        
        self.start_date = datetime.strptime(str(opt['start_date']), '%Y%m%d')
        self.end_date = datetime.strptime(str(opt['end_date']), '%Y%m%d')
        # self.length = (self.end_date - self.start_date).days + 1
        self.length = cesm_dt2idx(self.end_date) - cesm_dt2idx(self.start_date) + 1
        print('length: ', self.length)
        
        print('Loading CESM data')
        cesm_data_cnt = cesm_dt2idx(self.end_date) - cesm_dt2idx(self.start_date) + 1
        self.cesm = torch.Tensor(self.length, 384, 320)
        # cesm_start_date = datetime(1850, 1, 1)
        cesm_file = osp.join(self.lq_folder, 'b.e11.BLMTRC5CN.f19_g16.002.pop.h.nday1.SST.18500101-20051231.nc')
        cesm_data = np.ma.MaskedArray(nc.Dataset(cesm_file)['SST'][cesm_dt2idx(self.start_date):cesm_dt2idx(self.end_date) + 1, ::-1, :])
        cesm_data = np.roll(cesm_data, -35, axis=2) + 273.15 # shift the data to the left by 35 grids and convert to Kelvin
        self.cesm_mask = torch.from_numpy(1 - np.ma.getmask(cesm_data).astype(np.float32)[0])
        cesm_data = (cesm_data - self.mean) / self.std
        cesm_data = np.array(np.ma.filled(cesm_data, 0))
        self.cesm = torch.from_numpy(cesm_data)
        self.cesm = self.cesm.unsqueeze(1) # add channel dimension
        del cesm_data
        
        era5_data_cnt = 0
        self.era5 = torch.Tensor(self.length, 721, 1440)
        loop = tqdm(range(self.start_date.year, self.end_date.year + 1), desc='Loading ERA5 data')
        self.era5_mask = None
        for year in loop:
            for month in range(1, 13):
                date = datetime(year, month, 1)
                if date < self.start_date or date > self.end_date:
                    continue
                # era5_file shape: (30, 721, 1440)
                era5_file = osp.join(self.gt_folder, f'era5_sst_{date.year}_{date.month:02d}.npy')
                era5_data = np.load(era5_file)
                if is_leap_year(date.year) and date.month == 2:
                    era5_data = era5_data[:-1]
                era5_data = (era5_data - self.mean) / self.std
                if self.era5_mask is None:
                    self.era5_mask = torch.from_numpy(1 - np.isnan(era5_data).astype(np.float32)[0])
                # set all np.nan to 0
                era5_data = np.array(np.nan_to_num(era5_data))
                # print('era5_data: ', era5_data.shape)
                era5_idx = cesm_dt2idx(date) - cesm_dt2idx(self.start_date) 
                self.era5[era5_idx: era5_idx + era5_data.shape[0]] = torch.from_numpy(era5_data)
                era5_data_cnt += era5_data.shape[0]
                
        self.era5 = self.era5.unsqueeze(1) # add channel dimension
        
        assert era5_data_cnt == cesm_data_cnt, 'The number of days in ERA5 and CESM data are not equal'
        
        # self.cesm = (self.cesm - self.mean) / self.std
        # self.era5 = (self.era5 - self.mean) / self.std
        
            

    def __getitem__(self, index):
        lq_img = self.cesm[index]
        gt_img = self.era5[index]
        lq_mask = self.cesm_mask
        gt_mask = self.era5_mask
        
        return {'lq': lq_img, 'gt': gt_img, 'lq_mask': lq_mask, 'gt_mask': gt_mask, 'lq_path': 'cesm', 'gt_path': 'era5'}

    def __len__(self):
        return self.length
    
if __name__ == '__main__':
    # name: t2m_train
    # type: t2mDataset
    # dataroot_gt: /mnt/ssd/sr/datasets/t2m/y
    # dataroot_lq: /mnt/ssd/sr/datasets/t2m/x
    # start_date: 19810101
    # end_date: 20011231
    # mean: 16.6803827048836
    # std: 10.966654158164557
    # io_backend:
    #   type: disk
    opt = {
        'dataroot_gt': '/mnt/ssd/sr/datasets/sst_1940_1950/y',
        'dataroot_lq': '/mnt/ssd/sr/datasets/sst_1940_1950/x',
        'start_date': 19400101,
        'end_date': 19401231,
        'mean': 289.83038270488356,
        'std': 10.966654158164557,
        'io_backend': {
            'type': 'disk'
        }
    }
    
    dataset = sstDataset(opt)
    lq = dataset[0]['lq']
    gt = dataset[0]['gt']
    
    print('lq: ', lq.shape)
    print('gt: ', gt.shape)
    
    plt.imshow(lq[0])
    plt.colorbar()
    plt.savefig('lq.png')
    plt.close()
    
    plt.imshow(gt[0])
    plt.colorbar()
    plt.savefig('gt.png')
    plt.close()