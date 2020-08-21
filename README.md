# BasicSR

[GitHub](https://github.com/xinntao/BasicSR) | [Gitee码云](https://gitee.com/xinntao/BasicSR) <br>
[English](README.md) | [简体中文](README_CN.md)

BasicSR is an **open source** image and video super-resolution toolbox based on PyTorch (will extend to more restoration tasks in the future).<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>

## Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

Please run the following commands in the **BasicSR root path** to install BasicSR:<br>
Make sure that your GCC version: gcc >= 5.

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

## License

This project is released under the Apache 2.0 license.
More details are in [LICENSE](LICENSE/README.md).

#### Contact

If you have any question, please email `xintao.wang@outlook.com`.

<sub><sup>[BasicSR-private](https://github.com/xinntao/BasicSR-private)</sup></sub>
