# BasicSR [[ESRGAN]](https://github.com/xinntao/ESRGAN) [[SFTGAN]](https://github.com/xinntao/CVPR18-SFTGAN)
### :sun_with_face: We are updating codes and description these days.
An image super-resolution toolkit flexible for development. It now provides:

1. **PSNR-oriented SR** models (e.g., SRCNN, SRResNet and etc). You can try different architectures, e.g, ResNet Block, ResNeXt Block, Dense Block, Residual Dense Block, Poly Block, Dual Path Block, Squeeze-and-Excitation Block, Residual-in-Residual Dense Block and etc.
<!--   1. want to compare more structures for SR. e.g. ResNet Block, ResNeXt Block, Dense Block, Residual Dense Block, Poly Block, Dual Path Block, Squeeze-and-Excitation Block and etc.
   1. want to provide some useful tricks for training SR networks.
   1. We are also curious to know what is the upper bound of PSNR for bicubic downsampling kernel by using an extremely large model.-->
2. [**Enhanced SRGAN**](https://github.com/xinntao/ESRGAN) model. It achieves consistently better visual quality with more realistic and natural textures than [SRGAN](https://arxiv.org/abs/1609.04802) and won the first place in the [PIRM2018-SR Challenge](https://www.pirm2018.org/PIRM-SR.html). For more details, please refer to [Paper](), [ESRGAN repo](https://github.com/xinntao/ESRGAN). (If you just want to test the model, [ESRGAN repo](https://github.com/xinntao/ESRGAN) provides simpler testing codes.)
<p align="center">
  <img height="350" src="https://github.com/xinntao/ESRGAN/blob/master/figures/baboon.jpg">
</p>

3. [**SFTGAN**](https://github.com/xinntao/CVPR18-SFTGAN) model. It adopts Spatial Feature Transform (SFT) to effectively incorporate other conditions/priors, like semantic prior for image SR, representing by segmentation probability maps. For more details, please refer to [Papaer](https://arxiv.org/abs/1804.02815), [SFTGAN repo](https://github.com/xinntao/CVPR18-SFTGAN).
<p align="center">
  <img height="220" src="https://github.com/xinntao/CVPR18-SFTGAN/blob/master/imgs/network_structure.png">
</p>

### BibTex
<!--
    @article{wang2018esrgan,
        author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Loy, Chen Change and Qiao, Yu and Tang, Xiaoou},
        title={ESRGAN: Enhanced super-resolution generative adversarial networks},
        journal={arXiv preprint arXiv:1809.00219},
        year={2018}
    }
-->

    @inproceedings{wang2018esrgan,
        author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Loy, Chen Change and Qiao, Yu and Tang, Xiaoou},
        title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
        booktitle = {European Conference on Computer Vision (ECCV) Workshops},
        year = {2018}
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

### Dependencies

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 0.4.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb scikit-image`
- [option] Python packages: [`pip install tensorflow tensorboard_logger`](https://github.com/xinntao/BasicSR/tree/master/codes/utils), for visualizing curves.

## Codes
We provide a detailed explaination of the **code framework** in [`./codes`](https://github.com/xinntao/BasicSR/tree/master/codes).
<p align="center">
   <img src="https://c1.staticflickr.com/2/1859/30513344578_801bc60a82_b.jpg" height="300">
</p>

We also provides:

<!--1. evaluation metric codes.-->
1. Some useful scripts, more details in [`./codes/scripts`](https://github.com/xinntao/BasicSR/tree/master/codes/scripts). 
1. [Wiki](https://github.com/xinntao/BasicSR/wiki), e.g., How to make high quality gif with full (true) color, Matlab bicubic imresize and etc.

## Usage 
### How to test

### How to train

## Datasets

<table>
  <tr>
    <th>Name</th>
    <th>Datasets</th>
    <th>Short Description</th>
    <th>Google Drive</th>
    <th>Baidu Drive</th>
  </tr>
  <tr>
    <td rowspan="3">Classical SR Training</td>
    <td>T91</td>
    <td><sub>Desc</sub></td>
    <td rowspan="9"><a href="https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing">Google Drive</a></td>
    <td rowspan="9"><a href="https://pan.baidu.com/s/18fJzAHIg8Zpkc-2seGRW4Q">Baidu Drive</a></td>
  </tr>
 <tr>
    <td>BSD200</td>
    <td><sub>Desc</sub></td>
  </tr>
  <tr>
    <td>General 100</td>
    <td><sub>Desc</sub></td>
  </tr>
  <tr>
    <td rowspan="6">Classical SR Testing</td>
    <td>Set5</td>
    <td><sub>Desc</sub></td>
  </tr>
  <tr>
    <td>Set14</td>
    <td><sub>Desc</sub></td>
  </tr>
  <tr>
    <td>BSD100</td>
    <td><sub>Desc</sub></td>
  </tr>
  <tr>
    <td>Urban100</td>
    <td><sub>Desc</sub></td>
  </tr>
  <tr>
    <td>manga109</td>
    <td><sub>Desc</sub></td>
  </tr>
  <tr>
    <td>historical</td>
    <td><sub>Desc</sub></td>
  </tr>
   
  <tr>
    <td rowspan="3">2K Resolution</td>
    <td>DIV2K</td>
    <td><sub>Desc</sub></td>
    <td rowspan="3"><a href="https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing">Google Drive</a></td>
    <td rowspan="3"><a href="https://pan.baidu.com/s/18fJzAHIg8Zpkc-2seGRW4Q">Baidu Drive</a></td>
  </tr>
 <tr>
    <td>Flickr2K</td>
    <td><sub>Desc</sub></td>
  </tr>
 <tr>
    <td>DF2K</td>
    <td><sub>Desc</sub></td>
  </tr>
  
  <tr>
    <td rowspan="2">OST</td>
    <td>OST Training</td>
    <td><sub>Desc</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing">Google Drive</a></td>
    <td rowspan="2"><a href="https://pan.baidu.com/s/18fJzAHIg8Zpkc-2seGRW4Q">Baidu Drive</a></td>
  </tr>
 <tr>
    <td>OST300</td>
    <td><sub>Desc</sub></td>
  </tr>
</table>


Currently, there is a new DIVerse 2K resolution high quality images for SR called **DIV2K**, which can be downloaded from [DIV2K offical page](https://data.vision.ee.ethz.ch/cvl/DIV2K/), or from [Baidu Drive](https://pan.baidu.com/s/1LUj90_skqlVw4rjRVeEoiw).

## Pretrained models
We provide some pretrained models. More details about the pretrained models, please see [`experiments/pretrained_models`](https://github.com/xinntao/BasicSR/tree/master/experiments/pretrained_models).

You can put the downloaded models in the `experiments/pretrained_models` folder.


<table>
  <tr>
    <th>Name</th>
    <th>Modeds</th>
    <th>Short Description</th>
    <th>Google Drive</th>
    <th>Baidu Drive</th>
  </tr>
  <tr>
    <th rowspan="2">ESRGAN</th>
    <td>RRDB_ESRGAN_x4.pth</td>
    <td><sub>final ESRGAN model we used in our paper</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY">Google Drive</a></td>
    <td rowspan="2"><a href="https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ)">Baidu Drive</a></td>
  </tr>
  <tr>
    <td>RRDB_PSNR_x4.pth</td>
    <td><sub>model with high PSNR performance</sub></td>
  </tr>
   
  <tr>
    <th rowspan="4">SFTGAN</th>
    <td>segmentation_OST_bic.pth</td>
     <td><sub> segmentation model</sub></td>
    <td rowspan="4"><a href="https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing">Google Drive</a></td>
    <td rowspan="4"><a href="">Baidu Drive</a></td>
  </tr>
  <tr>
    <td>sft_net_ini.pth</td>
    <td><sub>sft_net for initilization</sub></td>
  </tr>
  <tr>
    <td>sft_net_torch.pth</td>
    <td><sub>SFTGAN Torch version (paper)</sub></td>
  </tr>
  <tr>
    <td>SFTGAN_bicx4_noBN_OST_bg.pth</td>
    <td><sub>SFTGAN PyTorch version</sub></td>
  </tr>
  
  <tr>
    <td >SRGAN<sup>*1</sup></td>
    <td>SRGAN_bicx4_303_505.pth</td>
     <td><sub> SRGAN(with modification)</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing">Google Drive</a></td>
    <td><a href="">Baidu Drive</a></td>
  </tr>
  
  <tr>
    <td >SRResNet<sup>*2</sup></td>
    <td>SRGAN_bicx4_303_505.pth</td>
     <td><sub> SRGAN(with modification)</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing">Google Drive</a></td>
    <td><a href="">Baidu Drive</a></td>
  </tr>
</table>



---
## :satisfied: Image Viewer - [HandyViewer](https://github.com/xinntao/HandyViewer)
If you have trouble in comparing image details, may have a try for [HandyViewer](https://github.com/xinntao/HandyViewer) - an image viewer that you can switch image with a fixed zoom ratio.

---



## Acknowlegement

- Code architecture is inspired by [pytorch-cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- Thanks to *Wai Ho Kwok*, who contributes to the initial version.
