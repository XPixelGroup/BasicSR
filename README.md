# BasicSR [[ESRGAN]](https://github.com/xinntao/ESRGAN) [[SFTGAN]](https://github.com/xinntao/CVPR18-SFTGAN)
### :sun_with_face: We are updating codes and description these days.
An image super-resolution toolkit flexible for development. It now provides:

1. **PSNR-oriented SR** models (e.g., SRCNN, SRResNet and etc). You can try different architectures, e.g, ResNet Block, ResNeXt Block, Dense Block, Residual Dense Block, Poly Block, Dual Path Block, Squeeze-and-Excitation Block, Residual-in-Residual Dense Block and etc.
<!--   1. want to compare more structures for SR. e.g. ResNet Block, ResNeXt Block, Dense Block, Residual Dense Block, Poly Block, Dual Path Block, Squeeze-and-Excitation Block and etc.
   1. want to provide some useful tricks for training SR networks.
   1. We are also curious to know what is the upper bound of PSNR for bicubic downsampling kernel by using an extremely large model.-->
2. [**Enhanced SRGAN**](https://github.com/xinntao/ESRGAN) model. It achieves consistently better visual quality with more realistic and natural textures than [SRGAN](https://arxiv.org/abs/1609.04802) and won the first place in the [PIRM2018-SR Challenge](https://www.pirm2018.org/PIRM-SR.html). For more details, please refer to [Papaer](), [ESRGAN repo](https://github.com/xinntao/ESRGAN). (If you just want to test the model, [ESRGAN repo](https://github.com/xinntao/ESRGAN) provides simpler testing codes.)
<p align="center">
  <img height="350" src="https://github.com/xinntao/ESRGAN/blob/master/figures/baboon.jpg">
</p>

3. [**SFTGAN**](https://github.com/xinntao/CVPR18-SFTGAN) model. It adopts Spatial Feature Transform (SFT) to effectively incorporate other conditions/priors, like semantic prior for image SR representing by segmentation probability maps. For more details, please refer to [Papaer](https://arxiv.org/abs/1804.02815), [SFTGAN repo](https://github.com/xinntao/CVPR18-SFTGAN).
<p align="center">
  <img height="220" src="https://github.com/xinntao/CVPR18-SFTGAN/blob/master/imgs/network_structure.png">
</p>

### BibTex
    @article{wang2018esrgan,
        author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Loy, Chen Change and Qiao, Yu and Tang, Xiaoou},
        title={ESRGAN: Enhanced super-resolution generative adversarial networks},
        journal={arXiv preprint arXiv:1809.00219},
        year={2018}
    }
    @inproceedings{wang2018sftgan,
        author = {Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
        title = {Recovering realistic texture in image super-resolution by deep spatial feature transform},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2018}

<!-- ### Table of Contents
1. [Introduction](#introduction)
1. [Introduction](#introduction)
### Introduction
-->

## Codes
The **code framework** and **usage** are detailed explained in [`./codes`](https://github.com/xinntao/BasicSR/tree/master/codes)
<p align="center">
   <img src="https://c1.staticflickr.com/2/1859/30513344578_801bc60a82_b.jpg" height="300">
</p>

We also provides:

1. Useful scripts in [`./codes/scripts`](https://github.com/xinntao/BasicSR/tree/master/codes/scripts).
1. We use [tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) to  visualize training loss, validation PSNR and etc. Installation and usage in [`tb_logger`](https://github.com/xinntao/BasicSR/tree/master/tb_logger)
1. [Wiki](https://github.com/xinntao/BasicSR/wiki), e.g., How to make high quality gif with full (true) color, Matlab bicubic imresize
<!--1. evaluation metric codes.-->

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




---
## :satisfied: Image Viewer - [HandyViewer](https://github.com/xinntao/HandyViewer)
If you have trouble in comparing image details, may have a try for [HandyViewer](https://github.com/xinntao/HandyViewer) - an image viewer that you can switch image with a fixed zoom ratio.

---



## Acknowlegement

- Code architecture is inspired by [pytorch-cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- Thanks to *Wai Ho Kwok*, who contributes to the initial version.
