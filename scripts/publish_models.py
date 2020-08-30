import glob
import subprocess
import torch
from os import path as osp

paths = glob.glob('experiments/pretrained_models/*.pth')

for idx, path in enumerate(paths):
    print(f'{idx+1:03d}: Processing {path}')
    net = torch.load(path, map_location=torch.device('cpu'))
    basename = osp.basename(path)
    if 'params' not in net and 'params_ema' not in net:
        raise ValueError(f'Please check! Model {basename} does not '
                         f"have 'params'/'params_ema' key.")
    else:
        if '-' in basename:
            # check whether the sha is the latest
            old_sha = basename.split('-')[1].split('.')[0]
            new_sha = subprocess.check_output(['sha256sum', path]).decode()[:8]
            if old_sha != new_sha:
                final_file = path.split('-')[0] + f'-{new_sha}.pth'
                print(f'\t Save from {path} to {final_file}')
                subprocess.Popen(['mv', path, final_file])
        else:
            sha = subprocess.check_output(['sha256sum', path]).decode()[:8]
            final_file = path.split('.pth')[0] + f'-{sha}.pth'
            print(f'\t Save from {path} to {final_file}')
            subprocess.Popen(['mv', path, final_file])
