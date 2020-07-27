# BasicSR

[GitHub](https://github.com/xinntao/BasicSR) | [Gitee码云](https://gitee.com/xinntao/BasicSR) <br>
[English](README.md) | [简体中文](README_CN.md)

BasicSR 是一个基于 PyTorch 的开源图像视频超分辨率 (Super-Resolution) 工具箱 (之后或许会支持更多的 Restoration 任务).<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>

## 依赖和安装
- Python >= 3.7 (推荐使用 [Anaconda](https://www.anaconda.com/download/#linux) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

在BasicSR的根目录下运行以下命令:
```bash
python setup.py develop
pip install -r requirements.txt
```

注意: BasicSR 仅在 Ubuntu 下进行测试，或许不支持Windows. 可以在Windows下尝试[支持CUDA的Windows WSL](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) :-) (目前只有Fast ring的预览版系统可以安装).

## TODO 清单
参见 [project boards](https://github.com/xinntao/BasicSR/projects).

## 数据准备
参见 [DatasetPreparation_CN.md](docs/DatasetPreparation_CN.md).

## 训练和测试
参见 [TrainTest_CN.md](docs/TrainTest_CN.md).

## 模型库和基准
结果和预训练的模型在 [ModelZoo_CN.md](docs/ModelZoo_CN.md).

## 代码库的设计和约定
参见 [DesignConvention_CN.md](docs/DesignConvention_CN.md).

## 许可
本项目使用 Apache 2.0 license.
更多细节参见 [LICENSE](LICENSE/README.md).

#### 联系
若有任何问题, 请电邮 `xintao.wang@outlook.com`.

<sub><sup>[BasicSR-private](https://github.com/xinntao/BasicSR-private)</sup></sub>

