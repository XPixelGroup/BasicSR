import cv2
import glob
import numpy as np
import torch
from os import path as osp

from basicsr.models.archs.rrdbnet_arch import RRDBNet

# configurations
model_path = 'experiments/pretrained_models/ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'  # noqa: E501
device = torch.device('cuda')
# device = torch.device('cpu')
test_img_folder = 'datasets/Set14/LRbicx4'

# set up model
model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

for idx, path in enumerate(sorted(glob.glob(test_img_folder))):
    imgname = osp.splitext(osp.basename(path))[0]
    print(idx, imgname)
    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]],
                                        (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)
    # inference
    with torch.no_grad():
        output = model(img)
    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(f'results/{imgname}_ESRGAN.png', output)
