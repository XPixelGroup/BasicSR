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

:compass: [入群二维码](#e-mail-contact) &emsp;&emsp; [入群指南 (腾讯文档)](https://docs.qq.com/doc/DYXBSUmxOT0xBZ05u)

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

- :white_check_mark: Sep 2, 2021. Add **SwinIR training and testing** codes: [SwinIR](https://github.com/JingyunLiang/SwinIR) by [Jingyun Liang](https://github.com/JingyunLiang):+1:. More details are in [HOWTOs.md](docs/HOWTOs.md#how-to-train-swinir-sr)
- :white_check_mark: Aug 5, 2021. Add NIQE, which produces the same results as MATLAB (both are 5.7296 for tests/data/baboon.png).
- :white_check_mark: July 31, 2021. Add **bi-directional video super-resolution** codes: [**BasicVSR** and IconVSR](https://arxiv.org/abs/2012.02181).
- :white_check_mark: July 20, 2021. Add **dual-blind face restoration** codes: [HiFaceGAN](https://github.com/Lotayou/Face-Renovation) codes by [Lotayou](https://lotayou.github.io/).
- :white_check_mark: Nov 29, 2020. Add **ESRGAN** and **DFDNet** [colab demo](colab)
- :white_check_mark: Sep 8, 2020. Add **blind face restoration** inference codes: [DFDNet](https://github.com/csxmli2016/DFDNet).
- :white_check_mark: Aug 27, 2020. Add **StyleGAN2 training and testing** codes: [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch).

<details>
  <summary>More</summary>
<ul>
  <li> Sep 8, 2020. Add <b>blind face restoration</b> inference codes: <b>DFDNet</b>. <br> <i><font color="#DCDCDC">ECCV20: Blind Face Restoration via Deep Multi-scale Component Dictionaries</font></i> <br> <i><font color="#DCDCDC">Xiaoming Li, Chaofeng Chen, Shangchen Zhou, Xianhui Lin, Wangmeng Zuo and Lei Zhang</font></i> </li>
  <li> Aug 27, 2020. Add <b>StyleGAN2</b> training and testing codes. <br> <i><font color="#DCDCDC">CVPR20: Analyzing and Improving the Image Quality of StyleGAN</font></i> <br> <i><font color="#DCDCDC">Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen and Timo Aila</font></i> </li>
  <li>Aug 19, 2020. A <b>brand-new</b> BasicSR v1.0.0 online.</li>
</ul>
</details>

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

| GAN |  |  |  | | |
| :--- | :---:        |     :---:      | :--- | :---:        |     :---:      |
| StyleGAN2   | [Train](docs/HOWTOs.md#How-to-train-StyleGAN2) | [Inference](docs/HOWTOs.md#How-to-inference-StyleGAN2) | | | |
| **Face Restoration** |  |  |  | | |
| DFDNet | - | [Inference](docs/HOWTOs.md#How-to-inference-DFDNet) | | | |
| **Super Resolution** |  |  |  | | |
| ESRGAN | *TODO* | *TODO* | SRGAN | *TODO* | *TODO*|
| EDSR | *TODO* | *TODO* | SRResNet | *TODO* | *TODO*|
| RCAN | *TODO* | *TODO* | SwinIR  | [Train](docs/HOWTOs.md#how-to-train-swinir-sr) | [Inference](docs/HOWTOs.md#how-to-inference-swinir-sr)|
| EDVR | *TODO* | *TODO* | DUF | - | *TODO* |
| BasicVSR | *TODO* | *TODO* | TOF | - | *TODO* |
| **Deblurring** |  |  |  | | |
| DeblurGANv2 | - | *TODO* |  | | |
| **Denoise** |  |  |  | | |
| RIDNet | - | *TODO* | CBDNet | - | *TODO*|

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
**[Datasets.md](docs/Datasets.md)**&emsp;|&emsp;**[Models.md](docs/Models.md)**&emsp;|&emsp;**[Config.md](Config.md)**&emsp;|&emsp;**[Logging.md](docs/Logging.md)**

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
  year =         {2020}
}
```

> Xintao Wang, Ke Yu, Kelvin C.K. Chan, Chao Dong and Chen Change Loy. BasicSR: Open Source Image and Video Restoration Toolbox. https://github.com/xinntao/BasicSR, 2020.

## :e-mail: Contact

If you have any questions, please email `xintao.wang@outlook.com`.

<p align="center">
  <img src="https://user-images.githubusercontent.com/17445847/134879983-6f2d663b-16e7-49f2-97e1-7c53c8a5f71a.jpg"  height="300">  &emsp;  &emsp;
  <img src="https://user-images.githubusercontent.com/17445847/134880057-f08e3d3b-2ab1-4ae8-966d-5753fe1f402a.png"  height="300">
</p>
