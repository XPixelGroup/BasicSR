import os
import os.path
import glob
import cv2
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.utils

import models.modules.seg_arch as Arch
from data.util import imresize
import utils.util as util


# options
test_img_folder_name = 'samples'  # HR images
test_img_folder = '../data/' + test_img_folder_name
save_prob_path = '../data/' + test_img_folder_name + '_segprob'  # probability maps
save_byteimg_path = '../data/' + test_img_folder_name + '_byteimg'  # segmentation annotations
save_colorimg_path = '../data/' + test_img_folder_name + '_colorimg' # segmentaion color results

# make dirs
util.mkdirs([save_prob_path, save_byteimg_path, save_colorimg_path])

# load model
seg_model = Arch.OutdoorSceneSeg()
load_path = '../experiments/pretrained_models/OutdoorSceneSeg_bic.pth'
seg_model.load_state_dict(torch.load(load_path), strict=True)
seg_model.eval()
seg_model = seg_model.cuda()

# look_up table # RGB
lookup_table = torch.from_numpy(np.array([
    [153, 153, 153], # 0, background
    [0, 255, 255], # 1, sky
    [109, 158, 235], # 2, water
    [183, 225, 205], # 3, grass
    [153, 0, 255], # 4, mountain
    [17, 85, 204], # 5, building
    [106, 168, 79], # 6, plant
    [224, 102, 102], # 7, animal
    [255, 255, 255], # 8/255, void
    ])).float()
lookup_table /= 255

idx = 0
for path in glob.glob(test_img_folder + '/*'):
    idx += 1
    basename = os.path.basename(path)
    base = os.path.splitext(basename)[0]
    print(idx, base)
    # read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()  # BGR, [0, 255]

    # matlab imresize
    # the implementation is slower than matlab, can use matlab to generate first
    img_LR = imresize(img/255, 1/4, antialiasing=True)
    img = imresize(img_LR, 4, antialiasing=True) * 255

    img[0] -= 103.939
    img[1] -= 116.779
    img[2] -= 123.68
    img = img.unsqueeze(0)
    img = img.cuda()
    output = seg_model(Variable(img, volatile=True)).data.float().cpu()
    # prob
    torch.save(output, os.path.join(save_prob_path, base+'_bic.pth'))  # 1x8xHxW
    # byte img
    output = output.squeeze()
    _, argmax = torch.max(output, 0)
    argmax = argmax.squeeze().byte()
    cv2.imwrite(os.path.join(save_byteimg_path, base+'.png'), argmax.numpy())
    # color img
    im_h, im_w = argmax.size()
    color = torch.FloatTensor(3, im_h, im_w).fill_(0) # black
    for i in range(8):
        mask = torch.eq(argmax, i)
        color.select(0,0).masked_fill_(mask, lookup_table[i][0]) # R
        color.select(0,1).masked_fill_(mask, lookup_table[i][1]) # G
        color.select(0,2).masked_fill_(mask, lookup_table[i][2]) # B
    # void
    mask = torch.eq(argmax, 255)
    color.select(0,0).masked_fill_(mask, lookup_table[8][0]) # R
    color.select(0,1).masked_fill_(mask, lookup_table[8][1]) # G
    color.select(0,2).masked_fill_(mask, lookup_table[8][2]) # B
    torchvision.utils.save_image(color, os.path.join(save_colorimg_path, base+'.png'), padding=0, \
        normalize=False)
