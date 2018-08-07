# BasicSR

pytorch 0.4 version

**BasicSR** wants to provide some basic deep-learning based models for super-resolution, including:

1. PSNR-oriented SR models (e.g., SRCNN, VDSR, SRResNet and etc)
   1. want to compare more structures for SR. e.g. ResNet Block, ResNeXt Block, Dense Block, Residual Dense Block, Poly Block, Dual Path Block, Squeeze-and-Excitation Block and etc.
   1. want to provide some useful tricks for training SR networks.
   <!--1. We are also curious to know what is the upper bound of PSNR for bicubic downsampling kernel by using an extremely large model.-->
1. GAN-based models for more visual-pleasing performance, especially textures.
    1. [SRGAN](https://arxiv.org/abs/1609.04802)
    1. [SFTGAN](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/)

The codes will be explained in each folder with README.md and the exploration will be put in [Wiki](https://github.com/xinntao/BasicSR/wiki).

**Testing** and **Training** can be found in [`codes/README.md`](https://github.com/xinntao/BasicSR/tree/master/codes).

:sun_with_face:

- It now supports a framework to train and test PSNR-oriented SR models. And we will gradually train and compare other models and try other techniques. <!--(e.g., intermediate loss for large model). -->

- For SRGAN, we reproduce the results using DIV2K dataset (w/o BatchNorm in the generator).

<p align="center">
  <img src="https://c1.staticflickr.com/2/1730/27869068197_bf631fa9fc.jpg" height="400">
  <img src="https://c1.staticflickr.com/2/1735/27869206717_9fd4813c5e.jpg" height="400">
</p>

- For SFTGAN, we provide the training and testing codes.

Welcome to report bugs :stuck_out_tongue_winking_eye:  and welcome to contribute to this repo :stuck_out_tongue_winking_eye: . I am not expert at coding, but I will try to keep the codes tidy.

<!-- ### Table of Contents
1. [Introduction](#introduction)
1. [Introduction](#introduction)

### Introduction
-->

## Prerequisites

- Linux
- Python3
- Pytorch 0.4
- NVIDIA GPU + CUDA

## Datasets
There are some **classical SR datasets**, for example:
- training datasets: BSD200, T91, General100;
- testing datasets: Set5, Set14, urban100, BSD100, manga109, historical

You can download these classical SR datasets from [Google Drive](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/18fJzAHIg8Zpkc-2seGRW4Q).

Currently, there is a new DIVerse 2K resolution high quality images for SR called **DIV2K**, which can be downloaded from [DIV2K offical page](https://data.vision.ee.ethz.ch/cvl/DIV2K/), or from [Baidu Drive](https://pan.baidu.com/s/1LUj90_skqlVw4rjRVeEoiw).

## Pretrained models
Pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing). You can put them in `experiments/pretrained_models` folder.

More details about the pretrained models, please see [`experiments/pretrained_models`](https://github.com/xinntao/BasicSR/tree/master/experiments/pretrained_models).


<!--
## Getting Started
### How to test a model
1. prepare your data and pretrained model
    1. `SRResNet_bicx4_in3nf64nb16.pth` is provided in [**experiments/pretrained_models**](https://github.com/xinntao/BasicSR/tree/master/experiments/pretrained_models) and other pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1vg_baYuagOXEhpwQgu54lJOyU8u1DsMW?usp=sharing).
1. modify the corresponding testing json file in `options/test/test.json`
1. test the model with the command `python3 test.py -opt options/test/test.json`

### How to train a model
1. prepare your data, please see [`codes/data/README.md`](https://github.com/xinntao/BasicSR/tree/master/codes/data) (it's better to test whether the data is ok using `test_dataloader`)
1. modify the corresponding training json file in `options/train/xxx.json`
1. train the model with the command `python3 train.py -opt options/train/xxx.json`

For more training details, please see [`codes/README.md`](https://github.com/xinntao/BasicSR/tree/master/codes).

---
## :satisfied: Image Viewer - [HandyViewer](https://github.com/xinntao/HandyViewer)
If you have trouble in comparing image details, may have a try for [HandyViewer](https://github.com/xinntao/HandyViewer) - an image viewer that you can switch image with a fixed zoom ratio.

---

## Pretrained Models
### Qualitative results [PSNR/dB] of SRResNet (EDSR)
See more details in [**experiments/pretrained_models**](https://github.com/xinntao/BasicSR/tree/master/experiments/pretrained_models)

| Model | Scale | Channel | DIV2K<sup>2</sup> | Set5| Set14 | BSD100 | Urban100 |
|--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SRResNet_bicx2_in3nf64nb16<sup>1</sup> | 2 | RGB | 34.720<sup>3</sup> | 35.835 | 31.643 | | |
|  |   |   | 36.143<sup>3</sup> | 37.947 | 33.682 | | |
| SRResNet_bicx3_in3nf64nb16 | 3 | RGB | 31.019 | 32.442  |  28.499 | | |
|  |   |   | 32.449 | 34.428  | 30.371  | | |
| SRResNet_bicx4_in3nf64nb16 | 4 | RGB | 29.051 | 30.278 | 26.853 | | |
|  |   |   | 30.486 | 32.180 | 28.645 | | |
| SRResNet_bicx8_in3nf64nb16 | 8 | RGB | 25.429 | 25.357 | 23.348 | | |
|  |   |   | 26.885 | 27.070 | 24.996 | | |
| SRResNet_bicx2_in1nf64nb16 | 2 | Y | 35.870 | 37.864 | 33.581 | | |
| SRResNet_bicx3_in1nf64nb16 | 3 | Y | 32.182 | 34.263 | 30.186 | | |
| SRResNet_bicx4_in1nf64nb16 | 4 | Y | 30.224 | 32.038 | 28.494 | | |
| SRResNet_bicx8_in1nf64nb16 | 8 | Y | 26.660 | 26.621 | 24.804 | | |

<sup>1</sup> **bic**: MATLAB bicubic downsampling; **in3**: input has 3 channels; **nf64**: 64 feature maps; **nb16**: 16 residual blocks.

<sup>2</sup> DIV2K 0801 ~ 0900 validation images.

<sup>3</sup> The first row is evaluated on RGB channels, while the secone row is evaluated on Y channel (of YCbCr).
-->

## Acknowlegement

- Code architecture is inspired from [pytorch-cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- Thanks to *Wai Ho Kwok*, who develops the initial version.
