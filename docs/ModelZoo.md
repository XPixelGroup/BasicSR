# Model Zoo and Baselines

[English](ModelZoo.md) **|** [简体中文](ModelZoo_CN.md)

We provide:

1. Official models converted directly from official released models
1. Reproduced models with `BasicSR`. Pre-trained models and log examples are provided

You can put the downloaded models in the `experiments/pretrained_models` folder.

**[Download official pre-trained models]** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g))(https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing))

You can use the scrip to download pre-trained models from Google Drive.

```python
python scripts/download_pretrained_models.py ESRGAN
# method can be ESRGAN, EDVR, StyleGAN, EDSR, DUF, DFDNet, dlib
```

**[Download reproduced models and logs]** ([Google Drive](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing), [百度网盘](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ))

In addition, we upload the training process and curves in [wandb](https://www.wandb.com/).

**[Training curves in wandb](https://app.wandb.ai/xintao/basicsr)**

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="../assets/wandb.jpg" height="350">
</a></p>

#### Contents

1. [Image Super-Resolution](#Image-Super-Resolution)
    1. [Image SR Official Models](#Image-SR-Official-Models)
    1. [Image SR Reproduced Models](#Image-SR-Reproduced-Models)
1. [Video Super-Resolution](#Video-Super-Resolution)

## Image Super-Resolution

When evaluation:

- We crop `scale` border pixels in each border
- Evaluated on RGB channels

### Image SR Official Models

|Exp Name         | Set5 (PSNR/SSIM)     | Set14 (PSNR/SSIM)   |DIV2K100 (PSNR/SSIM)   |
| :------------- | :----------:    | :----------:   |:----------:   |
| EDSR_Mx2_f64b16_DIV2K_official-3ba7b086 | 35.7768 / 0.9442 | 31.4966 / 0.8939 | 34.6291 / 0.9373 |
| EDSR_Mx3_f64b16_DIV2K_official-6908f88a | 32.3597 / 0.903 | 28.3932 / 0.8096 | 30.9438 / 0.8737 |
| EDSR_Mx4_f64b16_DIV2K_official-0c287733 | 30.1821 / 0.8641 | 26.7528 / 0.7432 | 28.9679 / 0.8183 |
| EDSR_Lx2_f256b32_DIV2K_official-be38e77d | 35.9979 / 0.9454 | 31.8583 / 0.8971 | 35.0495 / 0.9407 |
| EDSR_Lx3_f256b32_DIV2K_official-3660f70d | 32.643 / 0.906 | 28.644 / 0.8152 | 31.28 / 0.8798 |
| EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f | 30.5499 / 0.8701 | 27.0011 / 0.7509 | 29.277 / 0.8266 |

### Image SR Reproduced Models

Experiment name conventions are in [Config.md](Config.md).

|Exp Name         | Set5 (PSNR/SSIM)     | Set14 (PSNR/SSIM)   |DIV2K100 (PSNR/SSIM)   |
| :------------- | :----------:    | :----------:   |:----------:   |
| 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb | 30.2468 / 0.8651 | 26.7817 / 0.7451 | 28.9967 / 0.8195 |
| 002_MSRResNet_x2_f64b16_DIV2K_1000k_B16G1_001pretrain_wandb | 35.7483 / 0.9442 | 31.5403 / 0.8937 |34.6699 / 0.9377|
| 003_MSRResNet_x3_f64b16_DIV2K_1000k_B16G1_001pretrain_wandb | 32.4038 / 0.9032| 28.4418 / 0.8106|30.9726 / 0.8743 |
| 004_MSRGAN_x4_f64b16_DIV2K_400k_B16G1_wandb | 28.0158 / 0.8087|24.7474 / 0.6623 | 26.6504 / 0.7462|
| | | | |
| 201_EDSR_Mx2_f64b16_DIV2K_300k_B16G1_wandb | 35.7395 / 0.944|31.4348 / 0.8934 |34.5798 / 0.937 |
| 202_EDSR_Mx3_f64b16_DIV2K_300k_B16G1_201pretrain_wandb|32.315 / 0.9026 |28.3866 / 0.8088 |30.9095 / 0.8731|
| 203_EDSR_Mx4_f64b16_DIV2K_300k_B16G1_201pretrain_wandb|30.1726 / 0.8641 |26.721 / 0.743 |28.9506 / 0.818|
| 204_EDSR_Lx2_f256b32_DIV2K_300k_B16G1_wandb | 35.9792 / 0.9453 | 31.7284 / 0.8959 | 34.9544 / 0.9399 |
| 205_EDSR_Lx3_f256b32_DIV2K_300k_B16G1_204pretrain_wandb | 32.6467 / 0.9057 | 28.6859 / 0.8152 | 31.2664 / 0.8793 |
| 206_EDSR_Lx4_f256b32_DIV2K_300k_B16G1_204pretrain_wandb | 30.4718 / 0.8695 | 26.9616 / 0.7502 | 29.2621 / 0.8265 |

## Video Super-Resolution

#### Evaluation

In the evaluation, we include all the input frames and do not crop any border pixels unless otherwise stated.<br/>
We do not use the self-ensemble (flip testing) strategy and any other post-processing methods.

## EDVR

**Name convention**<br/>
EDVR\_(training dataset)\_(track name)\_(model complexity)

- track name. There are four tracks in the NTIRE 2019 Challenges on Video Restoration and Enhancement:
    - **SR**: super-resolution with a fixed downsampling kernel (MATLAB bicubic downsampling kernel is frequently used). Most of the previous video SR methods focus on this setting.
    - **SRblur**: the inputs are also degraded with motion blur.
    - **deblur**: standard deblurring (motion blur).
    - **deblurcomp**: motion blur + video compression artifacts.
- model complexity
    - **L** (Large): # of channels = 128, # of back residual blocks = 40. This setting is used in our competition submission.
    - **M** (Moderate): # of channels = 64, # of back residual blocks = 10.

[Download Models from Google Drive](https://drive.google.com/open?id=1WfROVUqKOBS5gGvQzBfU1DNZ4XwPA3LD)

| Model name |[Test Set] PSNR/SSIM |
|:----------:|:----------:|
| EDVR_Vimeo90K_SR_L | [Vid4] (Y<sup>1</sup>) 27.35/0.8264 [[↓Results]](https://drive.google.com/open?id=14nozpSfe9kC12dVuJ9mspQH5ZqE4mT9K)<br/> (RGB) 25.83/0.8077|
| EDVR_REDS_SR_M | [REDS] (RGB) 30.53/0.8699 [[↓Results]](https://drive.google.com/open?id=1Mek3JIxkjJWjhZhH4qVwTXnRZutKUtC-)|
| EDVR_REDS_SR_L | [REDS] (RGB) 31.09/0.8800 [[↓Results]](https://drive.google.com/open?id=1h6E0QVZyJ5SBkcnYaT1puxYYPVbPsTLt)|
| EDVR_REDS_SRblur_L | [REDS] (RGB) 28.88/0.8361 [[↓Results]](https://drive.google.com/open?id=1-8MNkQuMVMz30UilB9m_d0SXicwFEPZH)|
| EDVR_REDS_deblur_L | [REDS] (RGB) 34.80/0.9487 [[↓Results]](https://drive.google.com/open?id=133wCHTwiiRzenOEoStNbFuZlCX8Jn2at)|
| EDVR_REDS_deblurcomp_L | [REDS] (RGB) 30.24/0.8567 [[↓Results]](https://drive.google.com/open?id=1VjC4fXBXy0uxI8Kwxh-ijj4PZkfsLuTX)  |

<sup>1</sup> Y or RGB denotes the evaluation on Y (luminance) or RGB channels.

#### Stage 2 models for the NTIRE19 Competition
[Download Models from Google Drive](https://drive.google.com/drive/folders/1PMoy1cKlIYWly6zY0tG2Q4YAH7V_HZns?usp=sharing)

| Model name |[Test Set] PSNR/SSIM |
|:----------:|:----------:|
| EDVR_REDS_SR_Stage2 | [REDS] (RGB) / [[↓Results]]()|
| EDVR_REDS_SRblur_Stage2 | [REDS] (RGB) / [[↓Results]]()|
| EDVR_REDS_deblur_Stage2 | [REDS] (RGB) / [[↓Results]]()|
| EDVR_REDS_deblurcomp_Stage2 | [REDS] (RGB) / [[↓Results]]()  |


## DUF
The models are converted from the [officially released models](https://github.com/yhjo09/VSR-DUF). <br/>
[Download Models from Google Drive](https://drive.google.com/open?id=1seY9nclMuwk_SpqKQhx1ItTcQShM5R50)

| Model name | [Test Set] PSNR/SSIM<sup>1</sup> | Official Results<sup>2</sup> |
|:----------:|:----------:|:----------:|
| DUF_x4_52L_official<sup>3</sup> | [Vid4] (Y<sup>4</sup>) 27.33/0.8319 [[↓Results]](https://drive.google.com/open?id=1U9xGhlDSpPPQvKN0BAzXfjUCvaFxwsQf)<br/> (RGB) 25.80/0.8138   | (Y) 27.33/0.8318 [[↓Results]](https://drive.google.com/open?id=1HUmf__cSL7td7J4cXo2wvbVb14Y8YG2j)<br/> (RGB) 25.79/0.8136 |
| DUF_x4_28L_official | [Vid4]  | |
| DUF_x4_16L_official | [Vid4]  | |
| DUF_x3_16L_official | [Vid4]  | |
| DUF_x2_16L_official | [Vid4]  | |

<sup>1</sup> We crop eight pixels near image boundary for DUF due to its severe boundary effects. <br/>
<sup>2</sup> The official results are obtained by running the official codes and models. <br/>
<sup>3</sup> Different from the official codes, where `zero padding` is used for border frames, we use `new_info` strategy. <br/>
<sup>4</sup> Y or RGB denotes the evaluation on Y (luminance) or RGB channels.

## TOF
The models are converted from the [officially released models](https://github.com/anchen1011/toflow).<br/>
[Download Models from Google Drive](https://drive.google.com/open?id=18kJcxPLeNK8e0kYEiwmsnu9wVmhdMFFG)

| Model name | [Test Set] PSNR/SSIM | Official Results<sup>1</sup> |
|:----------:|:----------:|:----------:|
| TOF_official<sup>2</sup> | [Vid4] (Y<sup>3</sup>) 25.86/0.7626 [[↓Results]](https://drive.google.com/open?id=1Xp5U6uZeM44ShzawfuW_E-NmQ30hk-Be)<br/> (RGB)  24.38/0.7403 | (Y) 25.89/0.7651 [[↓Results]](https://drive.google.com/open?id=1WY3CcdzbRhpvDi3aGc1jAhIbeC6GUrM8)<br/> (RGB)  24.41/0.7428 |

<sup>1</sup> The official results are obtained by running the official codes and models. Note that TOFlow does not provide a strategy for border frame recovery and we simply use a `replicate` strategy for border frames. <br/>
<sup>2</sup> The converted model has slightly different results, due to different implementation. And we use `new_info` strategy for border frames. <br/>
<sup>3</sup> Y or RGB denotes the evaluation on Y (luminance) or RGB channels.
