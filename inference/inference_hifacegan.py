import os
import torch
from collections import OrderedDict
# change config file for ablation study...
from basicsr.models.archs.hifacegan_options import test_options
from basicsr.models.archs.hifacegan_arch import HiFaceGAN
from basicsr.data.hifacegan_dataset import HiFaceGANDataset
from basicsr.utils.img_util import tensor2img
import numpy as np
import cv2
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
opt = test_options()
device = torch.device('cuda:%d' % opt.gpu_ids[0])
model = HiFaceGAN(opt)
ckpt_path = os.path.join(
    opt.checkpoints_dir, opt.name, opt.which_epoch + '_net_G.pth'
)
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt)
model.to(device)

###  [Lotayou]: Critical Bug from 20200218 
#   When model is set to eval mode, the generated image
# is not enhanced whatsoever, with almost 0 residual.
# Using training mode seems to resolve this issue.
#
#   This is a bug in legacy Pytorch which seems to be fixed: 
#   https://github.com/pytorch/pytorch/pull/12671
###
model.train()

loader = torch.utils.data.DataLoader(
    dataset = HiFaceGANDataset(opt),
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
)
save_path = os.path.join(opt.results_dir, opt.name)
os.makedirs(save_path, exist_ok=True)

    
for data in tqdm(loader):
    lr = data['label'].to(device)
    hr = data['image'].to(device)
    sr = model(lr)
    
    pack = [lr, sr]
    if opt.with_gt:
        pack.extend(hr)
    pack = torch.cat(pack, dim=3)
    image = tensor2img(pack)
    save_name = data['path'][0].split('/')[-1]
    cv2.imwrite(os.path.join(save_path, save_name), image)
    