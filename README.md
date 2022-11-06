<p align="center">
  <img src="assets/basicsr_xpixel_logo.png" height=120>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center">

[![LICENSE](https://img.shields.io/github/license/xinntao/basicsr.svg)](https://github.com/xinntao/BasicSR/blob/master/LICENSE.txt)
[![PyPI](https://img.shields.io/pypi/v/basicsr)](https://pypi.org/project/basicsr/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/xinntao/BasicSR.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/xinntao/BasicSR/context:python)
[![python lint](https://github.com/xinntao/BasicSR/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/BasicSR/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/publish-pip.yml)
[![gitee mirror](https://github.com/xinntao/BasicSR/actions/workflows/gitee-mirror.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/gitee-mirror.yml)

</div>

<div align="center">

⚡[**HowTo**](#-HOWTOs) **|** 🔧[**Installation**](docs/INSTALL.md) **|** 💻[**Training Commands**](docs/TrainTest.md) **|** 🐢[**DatasetPrepare**](docs/DatasetPreparation.md) **|** 🏰[**Model Zoo**](docs/ModelZoo.md)

📕[**中文解读文档**](https://github.com/XPixelGroup/BasicSR-docs) **|** 📊 [**Plot scripts**](scripts/plot) **|** 📝[Introduction](docs/introduction.md) **|** <a href="https://github.com/XPixelGroup/BasicSR/tree/master/colab"><img src="https://colab.research.google.com/assets/colab-badge.svg" height="18" alt="google colab logo"></a> **|** ⏳[TODO List](https://github.com/xinntao/BasicSR/projects) **|** ❓[FAQ](docs/FAQ.md)
</div>

🚀 We add [BasicSR-Examples](https://github.com/xinntao/BasicSR-examples), which provides guidance and templates of using BasicSR as a python package. 🚀 <br>
📢 **技术交流QQ群**：**320960100** &emsp; 入群答案：**互帮互助共同进步** <br>
🧭 [入群二维码](#-contact) (QQ、微信) &emsp;&emsp; [入群指南 (腾讯文档)](https://docs.qq.com/doc/DYXBSUmxOT0xBZ05u) <br>

---

BasicSR (**Basic** **S**uper **R**estoration) is an open-source **image and video restoration** toolbox based on PyTorch, such as super-resolution, denoise, deblurring, JPEG artifacts removal, *etc*.<br>
BasicSR (**Basic** **S**uper **R**estoration) 是一个基于 PyTorch 的开源 图像视频复原工具箱, 比如 超分辨率, 去噪, 去模糊, 去 JPEG 压缩噪声等.

🚩 **New Features/Updates**

- ✅ July 26, 2022. Add plot scripts 📊[Plot](scripts/plot).
- ✅ May 9, 2022. BasicSR joins [XPixel](http://xpixel.group/).
- ✅ Oct 5, 2021. Add **ECBSR training and testing** codes: [ECBSR](https://github.com/xindongzhang/ECBSR).
  > ACMMM21: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices
- ✅ Sep 2, 2021. Add **SwinIR training and testing** codes: [SwinIR](https://github.com/JingyunLiang/SwinIR) by [Jingyun Liang](https://github.com/JingyunLiang). More details are in [HOWTOs.md](docs/HOWTOs.md#how-to-train-swinir-sr)
- ✅ Aug 5, 2021. Add NIQE, which produces the same results as MATLAB (both are 5.7296 for tests/data/baboon.png).
- ✅ July 31, 2021. Add **bi-directional video super-resolution** codes: [**BasicVSR** and IconVSR](https://arxiv.org/abs/2012.02181).
  > CVPR21: BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond
- **[More](docs/history_updates.md)**

---

If BasicSR helps your research or work, please help to ⭐ this repo or recommend it to your friends. Thanks😊 <br>
Other recommended projects:<br>
▶️ [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for general image restoration<br>
▶️ [GFPGAN](https://github.com/TencentARC/GFPGAN): A practical algorithm for real-world face restoration <br>
▶️ [facexlib](https://github.com/xinntao/facexlib): A collection that provides useful face-relation functions.<br>
▶️ [HandyView](https://github.com/xinntao/HandyView): A PyQt5-based image viewer that is handy for view and comparison. <br>
▶️ [HandyFigure](https://github.com/xinntao/HandyFigure): Open source of paper figures <br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>
<sub>([HandyCrawler](https://github.com/xinntao/HandyCrawler), [HandyWriting](https://github.com/xinntao/HandyWriting))</sub>

---

## ⚡ HOWTOs

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

## ✨ **Projects that use BasicSR**

- [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for general image restoration
- [**GFPGAN**](https://github.com/TencentARC/GFPGAN): A practical algorithm for real-world face restoration

If you use `BasicSR` in your open-source projects, welcome to contact me (by [email](#-contact) or opening an issue/pull request). I will add your projects to the above list 😊

## 📜 License and Acknowledgement

This project is released under the [Apache 2.0 license](LICENSE.txt).<br>
More details about **license** and **acknowledgement** are in [LICENSE](LICENSE/README.md).

## 🌏 Citations

If BasicSR helps your research or work, please cite BasicSR.<br>
The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.

``` latex
@misc{basicsr,
  author =       {Xintao Wang and Liangbin Xie and Ke Yu and Kelvin C.K. Chan and Chen Change Loy and Chao Dong},
  title =        {{BasicSR}: Open Source Image and Video Restoration Toolbox},
  howpublished = {\url{https://github.com/XPixelGroup/BasicSR}},
  year =         {2022}
}
```

> Xintao Wang, Liangbin Xie, Ke Yu, Kelvin C.K. Chan, Chen Change Loy and Chao Dong. BasicSR: Open Source Image and Video Restoration Toolbox. <https://github.com/xinntao/BasicSR>, 2022.

## 📧 Contact

If you have any questions, please email `xintao.alpha@gmail.com`, `xintao.wang@outlook.com`.

<br>

- **QQ群**: 扫描左边二维码 或者 搜索QQ群号: 320960100   入群答案：互帮互助共同进步
- **微信群**: 我们的一群已经满500人啦，二群也超过200人了；进群可以添加 Liangbin 的个人微信 (右边二维码)，他会在空闲的时候拉大家入群~

<p align="center">
  <img src="https://user-images.githubusercontent.com/17445847/134879983-6f2d663b-16e7-49f2-97e1-7c53c8a5f71a.jpg"  height="300">  &emsp;
  <img src="https://user-images.githubusercontent.com/17445847/139572512-8e192aac-00fa-432b-ac8e-a33026b019df.png"  height="300">
</p>

![visitors](https://visitor-badge.glitch.me/badge?page_id=XPixelGroup/BasicSR) (start from 2022-11-06)
