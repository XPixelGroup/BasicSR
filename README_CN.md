<p align="center">
  <img src="assets/basicsr_xpixel_logo.png" height=120>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

[![LICENSE](https://img.shields.io/github/license/xinntao/basicsr.svg)](https://github.com/xinntao/BasicSR/blob/master/LICENSE.txt)
[![PyPI](https://img.shields.io/pypi/v/basicsr)](https://pypi.org/project/basicsr/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/xinntao/BasicSR.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/xinntao/BasicSR/context:python)
[![python lint](https://github.com/xinntao/BasicSR/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/BasicSR/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/publish-pip.yml)
[![gitee mirror](https://github.com/xinntao/BasicSR/actions/workflows/gitee-mirror.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/gitee-mirror.yml)

<!-- [English](README.md) **|** [简体中文](README_CN.md) &emsp; [GitHub](https://github.com/xinntao/BasicSR) **|** [Gitee码云](https://gitee.com/xinntao/BasicSR) -->

:rocket: 我们添加了 [BasicSR-Examples](https://github.com/xinntao/BasicSR-examples), 它提供了使用BasicSR的指南以及模板 (以python package的形式) :rocket:

:loudspeaker: **技术交流QQ群**：**320960100** &emsp; 入群答案：**互帮互助共同进步**

:compass: [入群二维码](#e-mail-%E8%81%94%E7%B3%BB) (QQ、微信)  &emsp;&emsp; [入群指南 (腾讯文档)](https://docs.qq.com/doc/DYXBSUmxOT0xBZ05u)

---

<a href="https://drive.google.com/drive/folders/1G_qcpvkT5ixmw5XoN6MupkOzcK1km625?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height="18" alt="google colab logo"></a> Google Colab: [GitHub Link](colab) **|** [Google Drive Link](https://drive.google.com/drive/folders/1G_qcpvkT5ixmw5XoN6MupkOzcK1km625?usp=sharing) <br>
:m: [模型库](docs/ModelZoo_CN.md): :arrow_double_down: 百度网盘: [预训练模型](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g) **|** [复现实验](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ)
:arrow_double_down: Google Drive: [Pretrained Models](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing) **|** [Reproduced Experiments](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing) <br>
:file_folder: [数据](docs/DatasetPreparation_CN.md): :arrow_double_down: [百度网盘](https://pan.baidu.com/s/1AZDcEAFwwc1OC3KCd7EDnQ) (提取码:basr) :arrow_double_down: [Google Drive](https://drive.google.com/drive/folders/1gt5eT293esqY0yr1Anbm36EdnxWW_5oH?usp=sharing) <br>
:chart_with_upwards_trend: [wandb的训练曲线](https://app.wandb.ai/xintao/basicsr) <br>
:computer: [训练和测试的命令](docs/TrainTest_CN.md) <br>
:zap: [HOWTOs](#zap-howtos)

---

BasicSR (**Basic** **S**uper **R**estoration) 是一个基于 PyTorch 的开源图像视频复原工具箱, 比如 超分辨率, 去噪, 去模糊, 去 JPEG 压缩噪声等.

:triangular_flag_on_post: **新的特性/更新**

- :white_check_mark: Oct 5, 2021. 添加 **ECBSR 训练和测试** 代码: [ECBSR](https://github.com/xindongzhang/ECBSR).
  > ACMMM21: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices
- :white_check_mark: Sep 2, 2021. 添加 **SwinIR 训练和测试** 代码: [SwinIR](https://github.com/JingyunLiang/SwinIR) by [Jingyun Liang](https://github.com/JingyunLiang). 更多内容参见 [HOWTOs.md](docs/HOWTOs.md#how-to-train-swinir-sr)
- :white_check_mark: Aug 5, 2021. 添加了NIQE， 它输出和MATLAB一样的结果 (both are 5.7296 for tests/data/baboon.png).
- :white_check_mark: July 31, 2021. Add **bi-directional video super-resolution** codes: [**BasicVSR** and IconVSR](https://arxiv.org/abs/2012.02181).
  > CVPR21: BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond
- **[更多](docs/history_updates.md)**

:sparkles: **使用 BasicSR 的项目**

- [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN): 通用图像复原的实用算法
- [**GFPGAN**](https://github.com/TencentARC/GFPGAN): 真实场景人脸复原的实用算法

如果你的开源项目中使用了`BasicSR`, 欢迎联系我 ([邮件](#e-mail-%E8%81%94%E7%B3%BB)或者开一个issue/pull request)。我会将你的开源项目添加到上面的列表中 :blush:

---

如果 BasicSR 对你有所帮助，欢迎 :star: 这个仓库或推荐给你的朋友。Thanks:blush: <br>
其他推荐的项目:<br>
:arrow_forward: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): 通用图像复原的实用算法<br>
:arrow_forward: [GFPGAN](https://github.com/TencentARC/GFPGAN): 真实场景人脸复原的实用算法<br>
:arrow_forward: [facexlib](https://github.com/xinntao/facexlib): 提供实用的人脸相关功能的集合<br>
:arrow_forward: [HandyView](https://github.com/xinntao/HandyView): 基于PyQt5的 方便的看图比图工具<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>
<sub>([HandyView](https://gitee.com/xinntao/HandyView), [HandyFigure](https://gitee.com/xinntao/HandyFigure), [HandyCrawler](https://gitee.com/xinntao/HandyCrawler), [HandyWriting](https://gitee.com/xinntao/HandyWriting))</sub>

---

## :zap: HOWTOs

我们提供了简单的流程来快速上手 训练/测试/推理 模型. 这些命令并不能涵盖所有用法, 更多的细节参见下面的部分.

| GAN                  |                                              |                                              |          |                                                |                                                        |
| :------------------- | :------------------------------------------: | :------------------------------------------: | :------- | :--------------------------------------------: | :----------------------------------------------------: |
| StyleGAN2            | [训练](docs/HOWTOs_CN.md#如何训练-StyleGAN2) | [测试](docs/HOWTOs_CN.md#如何测试-StyleGAN2) |          |                                                |                                                        |
| **Face Restoration** |                                              |                                              |          |                                                |                                                        |
| DFDNet               |                      -                       |  [测试](docs/HOWTOs_CN.md#如何测试-DFDNet)   |          |                                                |                                                        |
| **Super Resolution** |                                              |                                              |          |                                                |                                                        |
| ESRGAN               |                    *TODO*                    |                    *TODO*                    | SRGAN    |                     *TODO*                     |                         *TODO*                         |
| EDSR                 |                    *TODO*                    |                    *TODO*                    | SRResNet |                     *TODO*                     |                         *TODO*                         |
| RCAN                 |                    *TODO*                    |                    *TODO*                    | SwinIR   | [Train](docs/HOWTOs.md#how-to-train-swinir-sr) | [Inference](docs/HOWTOs.md#how-to-inference-swinir-sr) |
| EDVR                 |                    *TODO*                    |                    *TODO*                    | DUF      |                       -                        |                         *TODO*                         |
| BasicVSR             |                    *TODO*                    |                    *TODO*                    | TOF      |                       -                        |                         *TODO*                         |
| **Deblurring**       |                                              |                                              |          |                                                |                                                        |
| DeblurGANv2          |                      -                       |                    *TODO*                    |          |                                                |                                                        |
| **Denoise**          |                                              |                                              |          |                                                |                                                        |
| RIDNet               |                      -                       |                    *TODO*                    | CBDNet   |                       -                        |                         *TODO*                         |

## :wrench: 依赖和安装

For detailed instructions refer to [docs/INSTALL.md](docs/INSTALL.md).

## :hourglass_flowing_sand: TODO 清单

参见 [project boards](https://github.com/xinntao/BasicSR/projects).

## :turtle: 数据准备

- 数据准备步骤, 参见 **[DatasetPreparation_CN.md](docs/DatasetPreparation_CN.md)**.
- 目前支持的数据集 (`torch.utils.data.Dataset`类), 参见 [Datasets_CN.md](docs/Datasets_CN.md).

## :computer: 训练和测试

- **训练和测试的命令**, 参见 **[TrainTest_CN.md](docs/TrainTest_CN.md)**.
- **Options/Configs**配置文件的说明, 参见 [Config_CN.md](docs/Config_CN.md).
- **Logging**日志系统的说明, 参见 [Logging_CN.md](docs/Logging_CN.md).

## :european_castle: 模型库和基准

- 目前支持的模型描述, 参见 [Models_CN.md](docs/Models_CN.md).
- **预训练模型和log样例**, 参见 **[ModelZoo_CN.md](docs/ModelZoo_CN.md)**.
- 我们也在 [wandb](https://app.wandb.ai/xintao/basicsr) 上提供了**训练曲线**等:

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="./assets/wandb.jpg" height="280">
</a></p>

## :memo: 代码库的设计和约定

参见 [DesignConvention_CN.md](docs/DesignConvention_CN.md).<br>
下图概括了整体的框架. 每个模块更多的描述参见: <br>
**[Datasets_CN.md](docs/Datasets_CN.md)**&emsp;|&emsp;**[Models_CN.md](docs/Models_CN.md)**&emsp;|&emsp;**[Config_CN.md](docs/Config_CN.md)**&emsp;|&emsp;**[Logging_CN.md](docs/Logging_CN.md)**

![overall_structure](./assets/overall_structure.png)

## :scroll: 许可

本项目使用 Apache 2.0 license.<br>
更多关于**许可**和**致谢**, 请参见 [LICENSE](LICENSE/README.md).

## :earth_asia: 引用

如果 BasicSR 对你有帮助, 请引用BasicSR. <br>
下面是一个 BibTex 引用条目, 它需要 `url` LaTeX package.

``` latex
@misc{basicsr,
  author =       {Xintao Wang and Liangbin Xie and Ke Yu and Kelvin C.K. Chan and Chen Change Loy and Chao Dong},
  title =        {{BasicSR}: Open Source Image and Video Restoration Toolbox},
  howpublished = {\url{https://github.com/XPixelGroup/BasicSR}},
  year =         {2022}
}
```

> Xintao Wang, Liangbin Xie, Ke Yu, Kelvin C.K. Chan, Chen Change Loy and Chao Dong. BasicSR: Open Source Image and Video Restoration Toolbox. <https://github.com/xinntao/BasicSR>, 2022.

## :e-mail: 联系

若有任何问题, 请电邮 `xintao.alpha@gmail.com`, `xintao.wang@outlook.com`.

<br>

- **QQ群**: 扫描左边二维码 或者 搜索QQ群号: 320960100   入群答案：互帮互助共同进步
- **微信群**: 我们的群一已经满500人啦，进群二可以扫描中间的二维码；如果进群遇到问题，也可以添加 Liangbin 的个人微信 (右边二维码)，他会在空闲的时候拉大家入群~

<p align="center">
  <img src="https://user-images.githubusercontent.com/17445847/134879983-6f2d663b-16e7-49f2-97e1-7c53c8a5f71a.jpg"  height="300">  &emsp;
  <img src="https://user-images.githubusercontent.com/52127135/172553058-6cf32e10-2959-42dd-b26a-f802f09343b0.png"  height="300">  &emsp;
  <img src="https://user-images.githubusercontent.com/17445847/139572512-8e192aac-00fa-432b-ac8e-a33026b019df.png"  height="300">
</p>
