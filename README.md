# BasicSR

[GitHub](https://github.com/xinntao/BasicSR) | [Gitee码云](https://gitee.com/xinntao/BasicSR) <br>
[English](README.md) | [简体中文](README_CN.md)

BasicSR is an open source image and video super-resolution (may extend to more restoration tasksin the future) toolbox based on PyTorch.<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>

## Dependencies and Installation
- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

Please run the following commands in the BasicSR root path to install BasicSR:
```bash
python setup.py develop
pip install -r requirements.txt
```

Note that BasicSR is only tested in Ubuntu, and may be not suitable for Windows. You may try [Windows WSL with CUDA supports](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) :-) (It is now only available for insider build with Fast ring).

## TODO List
Please see [project boards](https://github.com/xinntao/BasicSR/projects).

## Dataset Preparation
Please refer to [Datasets.md](docs/Datasets.md) for more details.

## Train and Test
Please see [TrainTest.md](docs/TrainTest.md) for the basic usage, *i.e.,* training and testing.

## Model Zoo and Baselines
Results and pre-trained models are available in the [ModelZoo.md](docs/ModelZoo.md).

## Codebase Designs and Conventions
Please see [DesignConvention.md](docs/DesignConvention.md) for the designs and convetions of the BasicSR codebase.

## License
This project is released under the Apache 2.0 license.
More details are in [LICENSE](LICENSE/README.md).

#### Contact
If you have any question, please email `xintao.wang@outlook.com`.

<sub><sup>[BasicSR-private](https://github.com/xinntao/BasicSR-private)</sup></sub>
