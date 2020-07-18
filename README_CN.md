# BasicSR

[GitHub](https://github.com/xinntao/BasicSR) | [Gitee码云](https://gitee.com/xinntao/BasicSR) <br>
[English](README.md) | [简体中文](README_CN.md)

BasicSR 是一个基于 PyTorch 的开源图像视频超分辨率(Super-Resolution)工具箱(之后或许会支持更多的Restoration).<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>

## 依赖和安装

- Python >= 3.7 (推荐使用[Anaconda](https://www.anaconda.com/download/#linux) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

在BasicSR根目录下运行以下命令:
```bash
python setup.py develop
pip install requirements.txt
```

注意: BasicSR 仅在Ubuntu下进行测试，或许不支持Windows. 期待[支持CUDA的Windows WSL](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) (目前Fast ring的预览版可以安装) :-)

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
