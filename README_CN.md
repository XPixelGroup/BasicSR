# :rocket: BasicSR

[GitHub](https://github.com/xinntao/BasicSR) | [Gitee码云](https://gitee.com/xinntao/BasicSR) <br>
[English](README.md) | [简体中文](README_CN.md)

BasicSR 是一个基于 PyTorch 的**开源**图像视频超分辨率 (Super-Resolution) 工具箱 (之后会支持更多的 Restoration 任务).<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>

## :sparkles: 新的特性

- Sep 8, 2020. 添加 **盲人脸复原推理代码: [DFDNet](https://github.com/csxmli2016/DFDNet)**. 注意和官方代码有些微差异.
   > Blind Face Restoration via Deep Multi-scale Component Dictionaries <br>
   > Xiaoming Li, Chaofeng Chen, Shangchen Zhou, Xianhui Lin, Wangmeng Zuo and Lei Zhang <br>
   > European Conference on Computer Vision (ECCV), 2020
- Aug 27, 2020. 添加 **StyleGAN2  训练和测试** 代码: [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch).
   > Analyzing and Improving the Image Quality of StyleGAN <br>
   > Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen and Timo Aila <br>
   > Computer Vision and Pattern Recognition (CVPR), 2020

<details>
  <summary>更多</summary>
<ul>
  <li>Aug 19, 2020. 全新的 BasicSR v1.0.0 上线.</li>
</ul>
</details>

## :zap:HOWTOs

我们提供了简单的流程来快速上手 训练/测试/推理 模型. 这些命令并不能涵盖所有用法, 更多的细节参见下面的部分.

- :zap: [如何训练 StyleGAN2](docs/HOWTOs_CN.md#如何训练-StyleGAN2)
- :zap: [如何测试 StyleGAN2](docs/HOWTOs_CN.md#如何测试-StyleGAN2)
- :zap: [如何测试 DFDNet](docs/HOWTOs_CN.md#如何测试-DFDNet)

## 依赖和安装

- Python >= 3.7 (推荐使用 [Anaconda](https://www.anaconda.com/download/#linux) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

在BasicSR的**根目录**下运行以下命令:<br>
(确保 GCC 版本: gcc >= 5)

```bash
pip install -r requirements.txt
python setup.py develop
```

注意: BasicSR 仅在 Ubuntu 下进行测试，或许不支持Windows. 可以在Windows下尝试[支持CUDA的Windows WSL](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) :-) (目前只有Fast ring的预览版系统可以安装).

## TODO 清单

参见 [project boards](https://github.com/xinntao/BasicSR/projects).

## 数据准备

- 数据准备步骤, 参见 **[DatasetPreparation_CN.md](docs/DatasetPreparation_CN.md)**.
- 目前支持的数据集 (`torch.utils.data.Dataset`类), 参见 [Datasets_CN.md](docs/Datasets_CN.md).

## 训练和测试

- **训练和测试的命令**, 参见 **[TrainTest_CN.md](docs/TrainTest_CN.md)**.
- **Options/Configs**配置文件的说明, 参见 [Config_CN.md](docs/Config_CN.md).
- **Logging**日志系统的说明, 参见 [Logging_CN.md](docs/Logging_CN.md).

## 模型库和基准

**[下载官方提供的预训练模型](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing)** <br>
**[下载复现的模型和log](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing)**

- 目前支持的模型描述, 参见 [Models_CN.md](docs/Models_CN.md).
- **预训练模型和log样例**, 参见 **[ModelZoo_CN.md](docs/ModelZoo_CN.md)**.
- 我们也在 [wandb](https://app.wandb.ai/xintao/basicsr) 上提供了**训练曲线**等:

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="./assets/wandb.jpg" height="280">
</a></p>

## 代码库的设计和约定

参见 [DesignConvention_CN.md](docs/DesignConvention_CN.md).<br>
下图概括了整体的框架. 每个模块更多的描述参见: <br>
**[Datasets_CN.md](docs/Datasets_CN.md)**&emsp;|&emsp;**[Models_CN.md](docs/Models_CN.md)**&emsp;|&emsp;**[Config_CN.md](Config_CN.md)**&emsp;|&emsp;**[Logging_CN.md](docs/Logging_CN.md)**

![overall_structure](./assets/overall_structure.png)

## 许可

本项目使用 Apache 2.0 license.
更多细节参见 [LICENSE](LICENSE/README.md).

#### 联系

若有任何问题, 请电邮 `xintao.wang@outlook.com`.
