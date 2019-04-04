# BasicSR [[ESRGAN]](https://github.com/xinntao/ESRGAN) [[SFTGAN]](https://github.com/xinntao/SFTGAN)

:black_square_button: TODO

- [ ] Use *yaml* instead of *json* for configurations
- [ ] Support distributed training
- [ ] Simplify the network structure file
- [ ] Provide new pre-trained models
- [ ] Update the creating lmdb script to add image one by one

:triangular_flag_on_post: <small>Add saving and loading training state. When resuming training, just pass a option with the name `resume_state`, like , `"resume_state": "../experiments/debug_001_RRDB_PSNR_x4_DIV2K/training_state/200.state"`. </small>

:triangular_flag_on_post: <small>Use Python logging, and support PyTorch 1.0</small>

An image super-resolution toolkit flexible for development. It now provides:

1. **PSNR-oriented SR** models (e.g., SRCNN, SRResNet and etc). You can try different architectures, e.g, ResNet Block, ResNeXt Block, Dense Block, Residual Dense Block, Poly Block, Dual Path Block, Squeeze-and-Excitation Block, Residual-in-Residual Dense Block and etc.
<!--   1. want to compare more structures for SR. e.g. ResNet Block, ResNeXt Block, Dense Block, Residual Dense Block, Poly Block, Dual Path Block, Squeeze-and-Excitation Block and etc.
   1. want to provide some useful tricks for training SR networks.
   1. We are also curious to know what is the upper bound of PSNR for bicubic downsampling kernel by using an extremely large model.-->
2. [**Enhanced SRGAN**](https://github.com/xinntao/ESRGAN) model (It can also train the **SRGAN** model). Enhanced SRGAN achieves consistently better visual quality with more realistic and natural textures than [SRGAN](https://arxiv.org/abs/1609.04802) and won the first place in the [PIRM2018-SR Challenge](https://www.pirm2018.org/PIRM-SR.html). For more details, please refer to [Paper](https://arxiv.org/abs/1809.00219), [ESRGAN repo](https://github.com/xinntao/ESRGAN). (If you just want to test the model, [ESRGAN repo](https://github.com/xinntao/ESRGAN) provides simpler testing codes.)
<p align="center">
  <img height="350" src="https://github.com/xinntao/ESRGAN/blob/master/figures/baboon.jpg">
</p>

3. [**SFTGAN**](https://github.com/xinntao/CVPR18-SFTGAN) model. It adopts Spatial Feature Transform (SFT) to effectively incorporate other conditions/priors, like semantic prior for image SR, representing by segmentation probability maps. For more details, please refer to [Papaer](https://arxiv.org/abs/1804.02815), [SFTGAN repo](https://github.com/xinntao/CVPR18-SFTGAN).
<p align="center">
  <img height="220" src="https://github.com/xinntao/SFTGAN/blob/master/figures/network_structure.png">
</p>

### BibTex

    @InProceedings{wang2018esrgan,
        author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
        title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
        booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
        month = {September},
        year = {2018}
    }
    @InProceedings{wang2018sftgan,
        author = {Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
        title = {Recovering realistic texture in image super-resolution by deep spatial feature transform},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2018}
    }
    
## Table of Contents
1. [Dependencies](#dependencies)
1. [Codes](#codes)
1. [Usage](#usage)
1. [Datasets](#datasets)
1. [Pretrained models](#pretrained-models)

### Dependencies

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 0.4.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb`
- [option] Python packages: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.

# Codes
[`./codes`](https://github.com/xinntao/BasicSR/tree/master/codes). We provide a detailed explaination of the **code framework** in [`./codes`](https://github.com/xinntao/BasicSR/tree/master/codes).
<p align="center">
   <img src="https://github.com/xinntao/public_figures/blob/master/BasicSR/code_framework.png" height="300">
</p>

We also provides:

1. Some useful scripts. More details in [`./codes/scripts`](https://github.com/xinntao/BasicSR/tree/master/codes/scripts). 
1. [Evaluation codes](https://github.com/xinntao/BasicSR/tree/master/metrics), e.g., PSNR/SSIM metric.
1. [Wiki](https://github.com/xinntao/BasicSR/wiki), e.g., How to make high quality gif with full (true) color, Matlab bicubic imresize and etc.

# Usage
### Data and model preparation
The common **SR datasets** can be found in [Datasets](#datasets). Detailed data preparation can be seen in [`codes/data`](https://github.com/xinntao/BasicSR/tree/master/codes/data).

We provide **pretrained models** in [Pretrained models](#pretrained-models).

## How to Test
### Test ESRGAN (SRGAN) models
1. Modify the configuration file `options/test/test_esrgan.json` 
1. Run command: `python test.py -opt options/test/test_esrgan.json`

### Test SR models
1. Modify the configuration file `options/test/test_sr.json` 
1. Run command: `python test.py -opt options/test/test_sr.json`

### Test SFTGAN models
1. Obtain the segmentation probability maps: `python test_seg.py`
1. Run command: `python test_sftgan.py`

## How to Train
### Train ESRGAN (SRGAN) models
We use a PSNR-oriented pretrained SR model to initialize the parameters for better quality.

1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](https://github.com/xinntao/BasicSR/tree/master/codes/data) and [wiki (Faster IO speed)](https://github.com/xinntao/BasicSR/wiki/Faster-IO-speed). 
1. Prerapre the PSNR-oriented pretrained model. You can use the `RRDB_PSNR_x4.pth` as the pretrained model. 
1. Modify the configuration file  `options/train/train_esrgan.json`
1. Run command: `python train.py -opt options/train/train_esrgan.json`

### Train SR models
1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](https://github.com/xinntao/BasicSR/tree/master/codes/data). 
1. Modify the configuration file `options/train/train_sr.json`
1. Run command: `python train.py -opt options/train/train_sr.json`

### Train SFTGAN models 
*Pretraining is also important*. We use a PSNR-oriented pretrained SR model (trained on DIV2K) to initialize the SFTGAN model.

1. First prepare the segmentation probability maps for training data: run [`test_seg.py`](https://github.com/xinntao/BasicSR/blob/master/codes/test_seg.py). We provide a pretrained segmentation model for 7 outdoor categories in [Pretrained models](#pretrained-models). We use [Xiaoxiao Li's codes](https://github.com/lxx1991/caffe_mpi) to train our segmentation model and transfer it to a PyTorch model.
1. Put the images and segmentation probability maps in a folder as described in [`codes/data`](https://github.com/xinntao/BasicSR/tree/master/codes/data).
1. Transfer the pretrained model parameters to the SFTGAN model. 
    1. First train with `debug` mode and obtain a saved model.
    1. Run [`transfer_params_sft.py`](https://github.com/xinntao/BasicSR/blob/master/codes/scripts/transfer_params_sft.py) to initialize the model.
    1. We provide an initialized model named `sft_net_ini.pth` in [Pretrained models](#pretrained-models)
1. Modify the configuration file in `options/train/train_sftgan.json`
1. Run command: `python train.py -opt options/train/train_sftgan.json`

# Datasets
Several common SR datasets are list below. 

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
    <td><sub>91 images for training</sub></td>
    <td rowspan="9"><a href="https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing">Google Drive</a></td>
    <td rowspan="9"><a href="https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg">Baidu Drive</a></td>
  </tr>
 <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS200</a></td>
    <td><sub>A subset (train) of BSD500 for training</sub></td>
  </tr>
  <tr>
    <td><a href="http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html">General100</a></td>
    <td><sub>100 images for training</sub></td>
  </tr>
  <tr>
    <td rowspan="6">Classical SR Testing</td>
    <td>Set5</td>
    <td><sub>Set5 test dataset</sub></td>
  </tr>
  <tr>
    <td>Set14</td>
    <td><sub>Set14 test dataset</sub></td>
  </tr>
  <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS100</a></td>
    <td><sub>A subset (test) of BSD500 for testing</sub></td>
  </tr>
  <tr>
    <td><a href="https://sites.google.com/site/jbhuang0604/publications/struct_sr">urban100</a></td>
    <td><sub>100 building images for testing (regular structures)</sub></td>
  </tr>
  <tr>
    <td><a href="http://www.manga109.org/en/">manga109</a></td>
    <td><sub>109 images of Japanese manga for testing</sub></td>
  </tr>
  <tr>
    <td>historical</td>
    <td><sub>10 gray LR images without the ground-truth</sub></td>
  </tr>
   
  <tr>
    <td rowspan="3">2K Resolution</td>
    <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">DIV2K</a></td>
    <td><sub>proposed in <a href="http://www.vision.ee.ethz.ch/ntire17/">NTIRE17</a>(800 train and 100 validation)</sub></td>
    <td rowspan="3"><a href="https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW?usp=sharing">Google Drive</a></td>
    <td rowspan="3"><a href="https://pan.baidu.com/s/1CFIML6KfQVYGZSNFrhMXmA">Baidu Drive</a></td>
  </tr>
 <tr>
    <td><a href="https://github.com/LimBee/NTIRE2017">Flickr2K</a></td>
    <td><sub>2650 2K images from Flickr for training</sub></td>
  </tr>
 <tr>
    <td>DF2K</td>
    <td><sub>A merged training dataset of DIV2K and Flickr2K</sub></td>
  </tr>
  
  <tr>
    <td rowspan="2">OST (Outdoor Scenes)</td>
    <td>OST Training</td>
    <td><sub>7 categories images with rich textures</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/u/1/folders/1iZfzAxAwOpeutz27HC56_y5RNqnsPPKr">Google Drive</a></td>
    <td rowspan="2"><a href="https://pan.baidu.com/s/1neUq5tZ4yTnOEAntZpK_rQ#list/path=%2Fpublic%2FSFTGAN&parentPath=%2Fpublic">Baidu Drive</a></td>
  </tr>
 <tr>
    <td>OST300</td>
    <td><sub>300 test images of outdoor scences</sub></td>
  </tr>
  
  <tr>
    <td >PIRM</td>
    <td>PIRM</td>
    <td><sub>PIRM self-val, val, test datasets</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/folders/17FmdXu5t8wlKwt8extb_nQAdjxUOrb1O?usp=sharing">Google Drive</a></td>
    <td rowspan="2"><a href="https://pan.baidu.com/s/1gYv4tSJk_RVCbCq4B6UxNQ">Baidu Drive</a></td>
  </tr>
</table>

# Pretrained models
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
    <td>SRResNet_bicx4_in3nf64nb16.pth</td>
     <td><sub> SRResNet(with modification)</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing">Google Drive</a></td>
    <td><a href="">Baidu Drive</a></td>
  </tr>
</table>



---
## :satisfied: Image Viewer - [HandyViewer](https://github.com/xinntao/HandyViewer)
May try [HandyViewer](https://github.com/xinntao/HandyViewer) - an image viewer that you can switch image with a fixed zoom ratio, easy for comparing image details.

---



## Acknowledgement

- Code architecture is inspired by [pytorch-cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- Thanks to *Wai Ho Kwok*, who contributes to the initial version.
