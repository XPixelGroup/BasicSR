# BasicSR

[GitHub](https://github.com/xinntao/BasicSR) | [Gitee码云](https://gitee.com/xinntao/BasicSR) <br>
[English](README.md) | [简体中文](README_CN.md)

BasicSR is an open source image and video super-resolution (may extend to restoration in the future) toolbox based on PyTorch.<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>

## Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

Please run the following cmds in the BasicSR root path to install BasicSR:
```bash
python setup.py develop
pip install requirements.txt
```

Note that BasicSR has only been tested in Ubuntu, and may be not suitable for Windows. You may try [Windows WSL with CUDA supports](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) (It is only available for insider build with Fast ring).

## TODO List
Please see [project boards](https://github.com/xinntao/BasicSR/projects).

## Dataset Preparation
Please refer to [DATASETS.md](docs/DATASETS.md) for more details.

## Training and Testing
Please see [TrainingTesting.md](docs/TrainingTesting.md) for the basic usage, *i.e.,* training and testing.

## Model Zoo and Baselines
Results and pre-trained models are available in the [ModelZoo.md](docs/ModelZoo.md).

## License
This project is released under the Apache 2.0 license.

#### Contact
If you have any question, please email `xintao.wang@outlook.com`.

<sub><sup>[BasicSR-private](https://github.com/xinntao/BasicSR-private)</sup></sub>
