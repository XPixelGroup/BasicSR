import argparse
import cv2
import glob
import numpy as np
import os
import shutil
import torch

from basicsr.archs.basicvsr_arch import BasicVSR
from basicsr.utils.img_util import img2tensor, tensor2img


def cache_imgs(paths, device='cpu'):
    imgs = []
    imgnames = []
    for idx, path in enumerate(paths):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        img = cv2.imread(path).astype(np.float32) / 255.
        imgs.append(img)
        imgnames.append(imgname)

    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)
    imgs = imgs.unsqueeze(0).to(device)
    return imgs, imgnames


def inference(imgs, imgnames, model, args):
    # inference
    outputs = model(imgs)

    # save imgs
    outputs = outputs.squeeze()
    outputs = list(outputs)
    for output, imgname in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(args.output, f'{imgname}_BasicVSR.png'), output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/BasicVSR_REDS4.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='datasets/REDS4/000', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/BasicVSR', help='output folder')
    parser.add_argument('--max_size', type=int, default=80, help='max image size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    model = BasicVSR(num_feat=64, num_block=30)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)

    # extract images from video format files
    input = args.input
    ffmpeg = False
    if not os.path.isdir(input):
        ffmpeg = True
        video_name = os.path.splitext(os.path.split(args.input)[-1])[0]
        input = os.path.join('./BasicVSR_tmp', video_name)
        os.makedirs(os.path.join('./BasicVSR_tmp', video_name), exist_ok=True)
        os.system('ffmpeg -i ' + args.input + ' -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  ' + input + '/frame%08d.png')

    # load data and inference
    imgs_list = sorted(glob.glob(os.path.join(input, '*')))
    imgs_num = len(imgs_list)
    if len(imgs_list) <= args.max_size:  # too many images may cause CUDA out of memory
        imgs, imgnames = cache_imgs(imgs_list, device=device)
        with torch.no_grad():
            inference(imgs, imgnames, model, args)
    else:
        for idx in range(0, imgs_num, args.max_size):
            max_size = min(args.max_size, imgs_num - idx)
            imgs, imgnames = cache_imgs(imgs_list[idx:idx + max_size], device=device)
            with torch.no_grad():
                inference(imgs, imgnames, model, args)

    # delete ffmpeg output images
    if ffmpeg:
        shutil.rmtree(input)


if __name__ == '__main__':
    main()
