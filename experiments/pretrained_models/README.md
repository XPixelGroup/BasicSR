# Pretrained models

Pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing). You can put them in `experiments/pretrained_models` folder.

- **SRResNet_bicx4_in3nf64nb16**.pth: SRResNet x4 model, trained on DIV2K, w/o BN, bicubic downsampling.
- **SRGAN_bicx4_303_505**.pth: SRGAN x4 model, trained on DIV2K, w/o BN, bicubic downsampling.
- **segmentation_OST_bic**.pth: segmentation model for bicubiced images, outdoor scenes.
- **sft_net_torch**.pth: torch version of SFTGAN model.
- **sft_net_ini**.pth: initialized SFTGAN model, initializing the sr generator with SRGAN_bicx4_303_505 parameters.
- **SFTGAN_bicx4_noBN_OST_bg**.pth: SFTGAN model, trained on OST dataset and use DIV2K as background images, w/o BN, bicubic downsampling.



## SRResNet (EDSR)

Through experiments, we found that

- no batch normalization
- residual block style: Conv-ReLU-Conv

are the best network settings.

#### Qualitative results [PSNR/dB] 
`SRResNet_bicx4_in3nf64nb16.pth` is provided here and other pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1vg_baYuagOXEhpwQgu54lJOyU8u1DsMW?usp=sharing).

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
| SRResNet_bicx4_in1nf64nb16 | 4 | Y | 30.224 | 32.038<sup>4</sup> | 28.494 | | |
| SRResNet_bicx8_in1nf64nb16 | 8 | Y | 26.660 | 26.621 | 24.804 | | |

<sup>1</sup> **bic**: MATLAB bicubic downsampling; **in3**: input has 3 channels; **nf64**: 64 feature maps; **nb16**: 16 residual blocks.

<sup>2</sup> DIV2K 0801 ~ 0900 validation images.

<sup>3</sup> The first row is evaluated on RGB channels, while the secone row is evaluated on Y channel (of YCbCr).

<sup>4</sup> (31.901, 29.711)
