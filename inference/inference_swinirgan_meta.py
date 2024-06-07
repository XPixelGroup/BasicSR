# Modified from https://github.com/JingyunLiang/SwinIR
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F
import netCDF4 as nc
from datetime import datetime, timedelta
from tqdm import tqdm

from basicsr.archs.swinir_meta_arch import SwinIRMetaUpsample


def nc2nchw(nc_filepath, part=0):
    """ncfile to nchw tensor

    Args:
        nc_filepath (str): path to the nc file
        part (int, optional): part of nc file. 0 for 850 to 1850, 1 for 1850 to 2005. Defaults to 0.
    """
    if part == 0:
        start_date = datetime(850, 1, 1)
        end_date = datetime(1849, 12, 31)
        # length = (end_date - start_date).days + 1
        nc_filename = os.path.join(nc_filepath, 'b.e11.BLMTRC5CN.f19_g16.002.cam.h1.TREFHT.08500101-18491231.nc')
    else:
        start_date = datetime(1951, 1, 1)
        end_date = datetime(2005, 12, 31)
        # length = (end_date - start_date).days + 1
        nc_filename = os.path.join(nc_filepath, 'b.e11.BLMTRC5CN.f19_g16.002.cam.h1.TREFHT.18500101-20051231.nc')
    
    print('Loading CESM data')
    mean = 275.90152
    std = 23.808582
    
    # cesm = torch.Tensor(length, 96, 144)
    cesm_data_nc = nc.Dataset(nc_filename)
    print('cesm_data_nc: ', cesm_data_nc)
    cesm_data = np.array(cesm_data_nc['TREFHT'][:, ::-1, :]) # shape of cesm_data: (-1, 96, 144)
    print('cesm_data.shape (np): ', cesm_data.shape)
    cesm = torch.from_numpy((cesm_data[:] - mean) / std)
    cesm = cesm.unsqueeze(1) # add channel dimension
    
    return cesm

def days_per_year(year):
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        return 366
    else:
        return 365

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/ssd/sr/datasets/t2m_1951_2005_infer/x', help='input test image folder')
    parser.add_argument('--output', type=str, default='/mnt/ssd/sr/datasets/t2m_1951_2005_infer/y', help='output folder')
    parser.add_argument(
        '--task',
        type=str,
        default='real_sr',
        help='classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car')
    # dn: denoising; car: compression artifact removal
    # TODO: it now only supports sr, need to adapt to dn and jpeg_car
    # parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
    # parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    # parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    # parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    # parser.add_argument('--large_model', action='store_true', help='Use large model, only used for real image sr')
    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/zzy/zyzhao/super_resolution/BasicSR/experiments/pretrained_models/net_g_395000.pth')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(args)
    model.eval()
    model = model.to(device)

    
    window_size = 8
    # cesm0 = nc2nchw(args.input, part=0) # 8500101-18491231
    cesm = nc2nchw(args.input, part=1) # 18500101-20051231
    print("cesm.shape: ", cesm.shape)
    cur_date = datetime(1940, 1, 1)
    cur_year_data = np.zeros((400, 721, 1440))
    cur_size = 0
    start_idx = (cur_date - datetime(1850, 1, 1)).days
    cesm = cesm[start_idx:]
    # cesm = torch.cat((cesm0, cesm1), dim=0)
    print("cesm.shape: ", cesm.shape)
    # split into batches
    cesm = torch.split(cesm, args.batch_size, dim=0)
    # del cesm0, cesm1
    
    mean = 275.90152
    std = 23.808582
    
    for idx, im_batch in tqdm(enumerate(cesm), desc='Inference'):
        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            im_batch = im_batch.to(device)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = im_batch.size()

            output = model(im_batch)
            output = output.data.squeeze().float()
            # output = output * std + mean
            output = output.cpu().numpy() # shape: (batch_size, 721, 1440)

        cur_year_data[cur_size:cur_size + output.shape[0]] = output
        cur_size += output.shape[0]
        
        days_in_cur_year = days_per_year(cur_date.year)
        if cur_size >= days_in_cur_year:
            # save annual data
            np.save(os.path.join(args.output, f'cesm_sr_t2m_{cur_date.year}.npy'), cur_year_data[:days_in_cur_year])
            print(f'Saved cesm_sr_t2m_{cur_date.year}.npy')
            remain_length = cur_size - days_in_cur_year
            cur_year_data[:remain_length] = cur_year_data[days_in_cur_year:days_in_cur_year + remain_length]
            cur_size = remain_length
        
        cur_date += timedelta(days=args.batch_size)
        
        if cur_date.year >= 1951:
            break


def define_model(args):
    model = SwinIRMetaUpsample(
        upscale_v=7.510416666666667,
        upscale_h=10,
        in_chans=1,
        img_size=(96, 144),
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=96,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='meta',
        resi_connection='1conv')

    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


if __name__ == '__main__':
    main()
