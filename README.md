# BasicSR

[GitHub](https://github.com/xinntao/BasicSR) | [Gitee码云](https://gitee.com/xinntao/BasicSR); [English](https://github.com/xinntao/BasicSR) | [简体中文](https://github.com/xinntao/BasicSR/blob/master/README_CN.md)

BasicSR is an open source image and video super-resolution toolbox based on PyTorch.<br>
([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))

## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

Please run the following cmds to install BasicSR.
```bash
python setup.py develop
pip install requirements.txt
```

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
If you have any question, please email `xintao.alpha@gmail.com`.

<sub><sup>[BasicSR-private](https://github.com/xinntao/BasicSR-private)</sup></sub>
