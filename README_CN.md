# :rocket: BasicSR

[English](README.md) **|** [简体中文](README_CN.md) &emsp; [GitHub](https://github.com/xinntao/BasicSR) **|** [Gitee码云](https://gitee.com/xinntao/BasicSR)

<a href="https://drive.google.com/drive/folders/1G_qcpvkT5ixmw5XoN6MupkOzcK1km625?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height="18" alt="google colab logo"></a> Google Colab: [GitHub Link](colab) **|** [Google Drive Link](https://drive.google.com/drive/folders/1G_qcpvkT5ixmw5XoN6MupkOzcK1km625?usp=sharing) <br>
:m: [模型库](docs/ModelZoo_CN.md) :arrow_double_down: 百度网盘: [预训练模型](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g) **|** [复现实验](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ)
:arrow_double_down: Google Drive: [Pretrained Models](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing) **|** [Reproduced Experiments](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing) <br>
:file_folder: [数据](docs/DatasetPreparation_CN.md) :arrow_double_down: [百度网盘](https://pan.baidu.com/s/1AZDcEAFwwc1OC3KCd7EDnQ) (提取码:basr) :arrow_double_down: [Google Drive](https://drive.google.com/drive/folders/1gt5eT293esqY0yr1Anbm36EdnxWW_5oH?usp=sharing) <br>
:chart_with_upwards_trend: [wandb的训练曲线](https://app.wandb.ai/xintao/basicsr) <br>
:computer: [训练和测试的命令](docs/TrainTest_CN.md) <br>
:zap: [HOWTOs](#zap-howtos)

---

BasicSR (**Basic** **S**uper **R**estoration) 是一个基于 PyTorch 的开源图像视频复原工具箱, 比如 超分辨率, 去噪, 去模糊, 去 JPEG 压缩噪声等.<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>
<sub>([HandyView](https://gitee.com/xinntao/HandyView), [HandyFigure](https://gitee.com/xinntao/HandyFigure), [HandyCrawler](https://gitee.com/xinntao/HandyCrawler), [HandyWriting](https://gitee.com/xinntao/HandyWriting))</sub>

## :sparkles: 新的特性

- Nov 29, 2020. 添加 **ESRGAN** and **DFDNet** [colab demo](colab).
- Sep 8, 2020. 添加 **盲人脸复原**测试代码: [DFDNet](https://github.com/csxmli2016/DFDNet).
- Aug 27, 2020. 添加 **StyleGAN2 训练和测试** 代码: [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch).

<details>
  <summary>更多</summary>
<ul>
  <li> Sep 8, 2020. 添加 <b>盲人脸复原</b> 测试代码: <b>DFDNet</b>. <br> <i><font color="#DCDCDC">ECCV20: Blind Face Restoration via Deep Multi-scale Component Dictionaries</font></i> <br> <i><font color="#DCDCDC">Xiaoming Li, Chaofeng Chen, Shangchen Zhou, Xianhui Lin, Wangmeng Zuo and Lei Zhang</font></i> </li>
  <li> Aug 27, 2020. 添加 <b>StyleGAN2</b> 训练和测试代码. <br> <i><font color="#DCDCDC">CVPR20: Analyzing and Improving the Image Quality of StyleGAN</font></i> <br> <i><font color="#DCDCDC">Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen and Timo Aila</font></i> </li>
  <li>Aug 19, 2020. <b>全新的</b> BasicSR v1.0.0 上线.</li>
</ul>
</details>

## :zap: HOWTOs

我们提供了简单的流程来快速上手 训练/测试/推理 模型. 这些命令并不能涵盖所有用法, 更多的细节参见下面的部分.

| GAN |  |  |  | | |
| :--- | :---:        |     :---:      | :--- | :---:        |     :---:      |
| StyleGAN2   | [训练](docs/HOWTOs_CN.md#如何训练-StyleGAN2) | [测试](docs/HOWTOs_CN.md#如何测试-StyleGAN2) | | | |
| **Face Restoration** |  |  |  | | |
| DFDNet | - | [测试](docs/HOWTOs_CN.md#如何测试-DFDNet) | | | |
| **Super Resolution** |  |  |  | | |
| ESRGAN | *TODO* | *TODO* | SRGAN | *TODO* | *TODO*|
| EDSR | *TODO* | *TODO* | SRResNet | *TODO* | *TODO*|
| RCAN | *TODO* | *TODO* |  |  | |
| EDVR | *TODO* | *TODO* | DUF | - | *TODO* |
| BasicVSR | *TODO* | *TODO* | TOF | - | *TODO* |
| **Deblurring** |  |  |  | | |
| DeblurGANv2 | - | *TODO* |  | | |
| **Denoise** |  |  |  | | |
| RIDNet | - | *TODO* | CBDNet | - | *TODO*|

## :wrench: 依赖和安装

- Python >= 3.7 (推荐使用 [Anaconda](https://www.anaconda.com/download/#linux) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### 通过 pip 安装

```bash
pip install basicsr
```

- 通过 pip 安装不会编译 cuda 算子.
- 如果你想使用 cuda 算子, 则需要设置环境变量 `BASICSR_JIT=True`. 注意: 通过这种方式, 在每一次运行模型时候, 它都需要编译, (比较耗时).
  - 例子: StyleGAN2 inference colab.

### 通过 git clone 然后 编译

1. Clone repo

    ```bash
    git clone https://github.com/xinntao/BasicSR.git
    ```

1. 安装依赖包

    ```bash
    cd BasicSR
    pip install -r requirements.txt
    ```

1. 安装 BasicSR

    在BasicSR的**根目录**下运行以下命令:<br>
    (确保 GCC 版本: gcc >= 5) <br>
    如果你需要以下 cuda 扩展算子: <br>
    &emsp;[*dcn* for EDVR](basicsr/ops)<br>
    &emsp;[*upfirdn2d* and *fused_act* for StyleGAN2](basicsr/ops)<br>
    在安装命令后添加 `--cuda_ext`.<br>
    如果使用 EDVR 和 StyleGAN2 模型, 需要使用上面的 cuda 扩展算子.

    ```bash
    python setup.py develop --cuda_ext
    ```

    否则, 安装的时候不会编译 cuda 算子.

    ```bash
    python setup.py develop
    ```

    你或许需要指定 CUDA 路径:

      ```bash
      CUDA_HOME=/usr/local/cuda \
      CUDNN_INCLUDE_DIR=/usr/local/cuda \
      CUDNN_LIB_DIR=/usr/local/cuda \
      python setup.py develop
      ```

注意: BasicSR 仅在 Ubuntu 下进行测试，或许不支持Windows. 可以在Windows下尝试[支持CUDA的Windows WSL](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) :-) (目前只有Fast ring的预览版系统可以安装).

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
**[Datasets_CN.md](docs/Datasets_CN.md)**&emsp;|&emsp;**[Models_CN.md](docs/Models_CN.md)**&emsp;|&emsp;**[Config_CN.md](Config_CN.md)**&emsp;|&emsp;**[Logging_CN.md](docs/Logging_CN.md)**

![overall_structure](./assets/overall_structure.png)

## :scroll: 许可

本项目使用 Apache 2.0 license.<br>
更多关于**许可**和**致谢**, 请参见 [LICENSE](LICENSE/README.md).

## :earth_asia: 引用

如果 BasicSR 对你有所帮助, 可以考虑引用BasicSR. <br>
下面是一个 BibTex 引用条目, 它需要 `url` LaTeX package.

``` latex
@misc{wang2020basicsr,
  author =       {Xintao Wang and Ke Yu and Kelvin C.K. Chan and
                  Chao Dong and Chen Change Loy},
  title =        {BasicSR},
  howpublished = {\url{https://github.com/xinntao/BasicSR}},
  year =         {2020}
}
```

> Xintao Wang, Ke Yu, Kelvin C.K. Chan, Chao Dong and Chen Change Loy. BasicSR. https://github.com/xinntao/BasicSR, 2020.

## :e-mail: 联系

若有任何问题, 请电邮 `xintao.wang@outlook.com`.
