import glob
import os.path as osp
import subprocess

import torch

paths = glob.glob('experiments/pretrained_models/*.pth')

for path in paths:
    model_basename = osp.basename(path).split('-')[0]
    net = torch.load(path)
    out_file = path.split('-')[0]
    torch.save(dict(params=net), out_file)

    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    final_file = out_file.rstrip('.pth') + f'-{sha[:8]}.pth'
    subprocess.Popen(['mv', out_file, final_file])
