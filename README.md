# :rocket: BasicSR

[GitHub](https://github.com/xinntao/BasicSR) | [Gitee码云](https://gitee.com/xinntao/BasicSR) <br>
[English](README.md) | [简体中文](README_CN.md)

BasicSR is an **open source** image and video super-resolution toolbox based on PyTorch (will extend to more restoration tasks in the future).<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>

## :sparkles: New Feature

- Sep 8, 2020. Add **blind face restoration inference codes: [DFDNet](https://github.com/csxmli2016/DFDNet)**. Note that it is slightly different from the official testing codes.
   > Blind Face Restoration via Deep Multi-scale Component Dictionaries <br>
   > Xiaoming Li, Chaofeng Chen, Shangchen Zhou, Xianhui Lin, Wangmeng Zuo and Lei Zhang <br>
   > European Conference on Computer Vision (ECCV), 2020
- Aug 27, 2020. Add **StyleGAN2 training and testing** codes: [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch).
   > Analyzing and Improving the Image Quality of StyleGAN <br>
   > Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen and Timo Aila <br>
   > Computer Vision and Pattern Recognition (CVPR), 2020

<details>
  <summary>More</summary>
<ul>
  <li>Aug 19, 2020. A brand-new BasicSR v1.0.0 online.</li>
</ul>
</details>

## :zap: HOWTOs

We provides simple pipelines to train/test/inference models for quick start.
These pipelines/commands cannot cover all the cases and more details are in the following sections.

- :zap: [How to train StyleGAN2](docs/HOWTOs.md#How-to-train-StyleGAN2)
- :zap: [How to test StyleGAN2](docs/HOWTOs.md#How-to-test-StyleGAN2)
- :zap: [How to test DFDNet](docs/HOWTOs.md#How-to-test-DFDNet)

## Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

Please run the following commands in the **BasicSR root path** to install BasicSR:<br>
(Make sure that your GCC version: gcc >= 5)

```bash
pip install -r requirements.txt
python setup.py develop
```

Note that BasicSR is only tested in Ubuntu, and may be not suitable for Windows. You may try [Windows WSL with CUDA supports](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) :-) (It is now only available for insider build with Fast ring).

## TODO List

Please see [project boards](https://github.com/xinntao/BasicSR/projects).

## Dataset Preparation

- Please refer to **[DatasetPreparation.md](docs/DatasetPreparation.md)** for more details.
- The descriptions of currently supported datasets (`torch.utils.data.Dataset` classes) are in [Datasets.md](docs/Datasets.md).

## Train and Test

- **Training and testing commands**: Please see **[TrainTest.md](docs/TrainTest.md)** for the basic usage.
- **Options/Configs**: Please refer to [Config.md](docs/Config.md).
- **Logging**: Please refer to [Logging.md](docs/Logging.md).

## Model Zoo and Baselines

**[Download official pre-trained models](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing)**<br>
**[Download reproduced models and logs](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing)**

- The descriptions of currently supported models are in [Models.md](docs/Models.md).
- **Pre-trained models and log examples** are available in **[ModelZoo.md](docs/ModelZoo.md)**.
- We also provide **training curves** in [wandb](https://app.wandb.ai/xintao/basicsr):

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="./assets/wandb.jpg" height="280">
</a></p>

## Codebase Designs and Conventions

Please see [DesignConvention.md](docs/DesignConvention.md) for the designs and conventions of the BasicSR codebase.<br>
The figure below shows the overall framework. More descriptions for each component: <br>
**[Datasets.md](docs/Datasets.md)**&emsp;|&emsp;**[Models.md](docs/Models.md)**&emsp;|&emsp;**[Config.md](Config.md)**&emsp;|&emsp;**[Logging.md](docs/Logging.md)**

![overall_structure](./assets/overall_structure.png)

## License and Acknowledgement

This project is released under the Apache 2.0 license.
More details about license and acknowledgement are in [LICENSE](LICENSE/README.md).

## Contact

If you have any question, please email `xintao.wang@outlook.com`.
