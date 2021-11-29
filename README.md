# :rocket: BasicSR

[![LICENSE](https://img.shields.io/github/license/xinntao/basicsr.svg)](https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/basicsr)](https://pypi.org/project/basicsr/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/xinntao/BasicSR.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/xinntao/BasicSR/context:python)
[![python lint](https://github.com/xinntao/BasicSR/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/BasicSR/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/publish-pip.yml)
[![gitee mirror](https://github.com/xinntao/BasicSR/actions/workflows/gitee-mirror.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/gitee-mirror.yml)

[English](README.md) **|** [简体中文](README_CN.md) &emsp; [GitHub](https://github.com/xinntao/BasicSR) **|** [Gitee码云](https://gitee.com/xinntao/BasicSR)

:rocket: We add [BasicSR-Examples](https://github.com/xinntao/BasicSR-examples), which provides guidance and templates of using BasicSR as a python package. :rocket:

:loudspeaker: **技术交流QQ群**：**320960100** &emsp; 入群答案：**互帮互助共同进步**

:compass: [入群二维码](#e-mail-contact) (QQ、微信) &emsp;&emsp; [入群指南 (腾讯文档)](https://docs.qq.com/doc/DYXBSUmxOT0xBZ05u)

---

<a href="https://drive.google.com/drive/folders/1G_qcpvkT5ixmw5XoN6MupkOzcK1km625?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height="18" alt="google colab logo"></a> Google Colab: [GitHub Link](colab) **|** [Google Drive Link](https://drive.google.com/drive/folders/1G_qcpvkT5ixmw5XoN6MupkOzcK1km625?usp=sharing) <br>
:m: [Model Zoo](docs/ModelZoo.md): :arrow_double_down: Google Drive: [Pretrained Models](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing) **|** [Reproduced Experiments](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing)
:arrow_double_down: 百度网盘: [预训练模型](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g) **|** [复现实验](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ) <br>
:file_folder: [Datasets](docs/DatasetPreparation.md): :arrow_double_down: [Google Drive](https://drive.google.com/drive/folders/1gt5eT293esqY0yr1Anbm36EdnxWW_5oH?usp=sharing) :arrow_double_down: [百度网盘](https://pan.baidu.com/s/1AZDcEAFwwc1OC3KCd7EDnQ) (提取码:basr)<br>
:chart_with_upwards_trend: [Training curves in wandb](https://app.wandb.ai/xintao/basicsr) <br>
:computer: [Commands for training and testing](docs/TrainTest.md) <br>
:zap: [HOWTOs](#zap-howtos)

---

BasicSR (**Basic** **S**uper **R**estoration) is an open-source **image and video restoration** toolbox based on PyTorch, such as super-resolution, denoise, deblurring, JPEG artifacts removal, *etc*.

BasicSR (**Basic** **S**uper **R**estoration) 是一个基于 PyTorch 的开源 图像视频复原工具箱, 比如 超分辨率, 去噪, 去模糊, 去 JPEG 压缩噪声等.

:triangular_flag_on_post: **New Features/Updates**

- :white_check_mark: Oct 5, 2021. Add **ECBSR training and testing** codes: [ECBSR](https://github.com/xindongzhang/ECBSR).
  > ACMMM21: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices
- :white_check_mark: Sep 2, 2021. Add **SwinIR training and testing** codes: [SwinIR](https://github.com/JingyunLiang/SwinIR) by [Jingyun Liang](https://github.com/JingyunLiang). More details are in [HOWTOs.md](docs/HOWTOs.md#how-to-train-swinir-sr)
- :white_check_mark: Aug 5, 2021. Add NIQE, which produces the same results as MATLAB (both are 5.7296 for tests/data/baboon.png).
- :white_check_mark: July 31, 2021. Add **bi-directional video super-resolution** codes: [**BasicVSR** and IconVSR](https://arxiv.org/abs/2012.02181).
  > CVPR21: BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond
- **[More](docs/history_updates.md)**

:sparkles: **Projects that use BasicSR**
- [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for general image restoration
- [**GFPGAN**](https://github.com/TencentARC/GFPGAN): A practical algorithm for real-world face restoration

If you use `BasicSR` in your open-source projects, welcome to contact me (by [email](#e-mail-contact) or opening an issue/pull request). I will add your projects to the above list :blush:

---

If BasicSR helps your research or work, please help to :star: this repo or recommend it to your friends. Thanks:blush: <br>
Other recommended projects:<br>
:arrow_forward: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for general image restoration<br>
:arrow_forward: [GFPGAN](https://github.com/TencentARC/GFPGAN): A practical algorithm for real-world face restoration <br>
:arrow_forward: [facexlib](https://github.com/xinntao/facexlib): A collection that provides useful face-relation functions.<br>
:arrow_forward: [HandyView](https://github.com/xinntao/HandyView): A PyQt5-based image viewer that is handy for view and comparison. <br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>
<sub>([HandyView](https://github.com/xinntao/HandyView), [HandyFigure](https://github.com/xinntao/HandyFigure), [HandyCrawler](https://github.com/xinntao/HandyCrawler), [HandyWriting](https://github.com/xinntao/HandyWriting))</sub>

---

## :zap: HOWTOs

We provide simple pipelines to train/test/inference models for a quick start.
These pipelines/commands cannot cover all the cases and more details are in the following sections.

| GAN                  |                                                |                                                        |          |                                                |                                                        |
| :------------------- | :--------------------------------------------: | :----------------------------------------------------: | :------- | :--------------------------------------------: | :----------------------------------------------------: |
| StyleGAN2            | [Train](docs/HOWTOs.md#How-to-train-StyleGAN2) | [Inference](docs/HOWTOs.md#How-to-inference-StyleGAN2) |          |                                                |                                                        |
| **Face Restoration** |                                                |                                                        |          |                                                |                                                        |
| DFDNet               |                       -                        |  [Inference](docs/HOWTOs.md#How-to-inference-DFDNet)   |          |                                                |                                                        |
| **Super Resolution** |                                                |                                                        |          |                                                |                                                        |
| ESRGAN               |                     *TODO*                     |                         *TODO*                         | SRGAN    |                     *TODO*                     |                         *TODO*                         |
| EDSR                 |                     *TODO*                     |                         *TODO*                         | SRResNet |                     *TODO*                     |                         *TODO*                         |
| RCAN                 |                     *TODO*                     |                         *TODO*                         | SwinIR   | [Train](docs/HOWTOs.md#how-to-train-swinir-sr) | [Inference](docs/HOWTOs.md#how-to-inference-swinir-sr) |
| EDVR                 |                     *TODO*                     |                         *TODO*                         | DUF      |                       -                        |                         *TODO*                         |
| BasicVSR             |                     *TODO*                     |                         *TODO*                         | TOF      |                       -                        |                         *TODO*                         |
| **Deblurring**       |                                                |                                                        |          |                                                |                                                        |
| DeblurGANv2          |                       -                        |                         *TODO*                         |          |                                                |                                                        |
| **Denoise**          |                                                |                                                        |          |                                                |                                                        |
| RIDNet               |                       -                        |                         *TODO*                         | CBDNet   |                       -                        |                         *TODO*                         |

## :wrench: Dependencies and Installation

For detailed instructions refer to [INSTALL.md](INSTALL.md).
## :hourglass_flowing_sand: TODO List

Please see [project boards](https://github.com/xinntao/BasicSR/projects).

## :turtle: Dataset Preparation

- Please refer to **[DatasetPreparation.md](docs/DatasetPreparation.md)** for more details.
- The descriptions of currently supported datasets (`torch.utils.data.Dataset` classes) are in [Datasets.md](docs/Datasets.md).

## :computer: Train and Test

- **Training and testing commands**: Please see **[TrainTest.md](docs/TrainTest.md)** for the basic usage.
- **Options/Configs**: Please refer to [Config.md](docs/Config.md).
- **Logging**: Please refer to [Logging.md](docs/Logging.md).

## :european_castle: Model Zoo and Baselines

- The descriptions of currently supported models are in [Models.md](docs/Models.md).
- **Pre-trained models and log examples** are available in **[ModelZoo.md](docs/ModelZoo.md)**.
- We also provide **training curves** in [wandb](https://app.wandb.ai/xintao/basicsr):

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="./assets/wandb.jpg" height="280">
</a></p>

## :memo: Codebase Designs and Conventions

Please see [DesignConvention.md](docs/DesignConvention.md) for the designs and conventions of the BasicSR codebase.<br>
The figure below shows the overall framework. More descriptions for each component: <br>
**[Datasets.md](docs/Datasets.md)**&emsp;|&emsp;**[Models.md](docs/Models.md)**&emsp;|&emsp;**[Config.md](docs/Config.md)**&emsp;|&emsp;**[Logging.md](docs/Logging.md)**

![overall_structure](./assets/overall_structure.png)

## :scroll: License and Acknowledgement

This project is released under the Apache 2.0 license.<br>
More details about **license** and **acknowledgement** are in [LICENSE](LICENSE/README.md).

## :earth_asia: Citations

If BasicSR helps your research or work, please consider citing BasicSR.<br>
The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.

``` latex
@misc{wang2020basicsr,
  author =       {Xintao Wang and Ke Yu and Kelvin C.K. Chan and
                  Chao Dong and Chen Change Loy},
  title =        {{BasicSR}: Open Source Image and Video Restoration Toolbox},
  howpublished = {\url{https://github.com/xinntao/BasicSR}},
  year =         {2018}
}
```

> Xintao Wang, Ke Yu, Kelvin C.K. Chan, Chao Dong and Chen Change Loy. BasicSR: Open Source Image and Video Restoration Toolbox. https://github.com/xinntao/BasicSR, 2018.


## :e-mail: Contact

If you have any questions, please email `xintao.wang@outlook.com`.

<br>

- **QQ群**: 扫描左边二维码 或者 搜索QQ群号: 320960100   入群答案：互帮互助共同进步
- **微信群**: 因为微信群超过200人，需要邀请才可以进群；要进微信群的小伙伴可以先添加 Liangbin 的个人微信 (右边二维码)，他会在空闲的时候拉大家入群~

<p align="center">
  <img src="https://user-images.githubusercontent.com/17445847/134879983-6f2d663b-16e7-49f2-97e1-7c53c8a5f71a.jpg"  height="300">  &emsp;  &emsp;
  <img src="https://user-images.githubusercontent.com/17445847/139572512-8e192aac-00fa-432b-ac8e-a33026b019df.png"  height="300">
</p>
