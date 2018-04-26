import os
import sys
import time
import random
import argparse
import numpy as np
from collections import OrderedDict

import torch
from torchvision.transforms import ToTensor

import options.options as option
import utils.util as util
import utils.metric as metric


# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
args = parser.parse_args()
options_path = args.opt
opt = option.parse(options_path, is_train=False)
util.mkdirs((path for key , path in opt['path'].items() if not key == 'pretrain_model_G')) #Make all directories needed
opt = option.dict_to_nonedict(opt)

# print to file and std_out simultaneously
class PrintLogger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(opt['path']['log'], 'print_log.txt'), "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
sys.stdout = PrintLogger()

from data import create_dataset, create_dataloader
from models import create_model

# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in opt['datasets'].items():
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    test_size = len(test_set)
    test_set_name = dataset_opt['name']
    print('Number of test images in [%s]: %d' % (test_set_name, test_size))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)
model.eval()

# Path for log file
test_log_path = os.path.join(opt['path']['log'], 'test_log.txt')
if os.path.exists(test_log_path):
    os.remove(test_log_path)
    print('Old test log is removed.')

print('Start Testing ...')

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    print('Testing [%s]...' % test_set_name)
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    for data in test_loader:
        need_HR = True
        if test_loader.dataset.opt['dataroot_HR'] is None:
            need_HR = False
        model.feed_data(data, volatile=True, need_HR=need_HR)
        img_path = data['LR_path'][0]

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(img_name)

        model.test()  # test
        visuals = model.get_current_visuals(need_HR=need_HR)

        # save intermediate numpy
        # t = visuals['super-resolution'].clone()
        # t = t.cpu().float().numpy()
        # np.save(os.path.join(dataset_dir, img_name), t)

        sr_img = util.tensor2img_np(visuals['SR']) # uint8

        # Save SR images for reference
        # exp_idx = opt.name.split('_')[0]
        save_img_path = os.path.join(dataset_dir, img_name+'.png')
        util.save_img_np(sr_img, save_img_path)
