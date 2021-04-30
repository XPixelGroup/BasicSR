# :rocket: BasicSR

[English](README.md) **|** [简体中文](README_CN.md) &emsp; [GitHub](https://github.com/xinntao/BasicSR) **|** [Gitee码云](https://gitee.com/xinntao/BasicSR)

<a href="https://drive.google.com/drive/folders/1G_qcpvkT5ixmw5XoN6MupkOzcK1km625?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height="18" alt="google colab logo"></a> Google Colab: [GitHub Link](colab) **|** [Google Drive Link](https://drive.google.com/drive/folders/1G_qcpvkT5ixmw5XoN6MupkOzcK1km625?usp=sharing) <br>
:m: [Model Zoo](docs/ModelZoo.md) :arrow_double_down: Google Drive: [Pretrained Models](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing) **|** [Reproduced Experiments](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing)
:arrow_double_down: 百度网盘: [预训练模型](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g) **|** [复现实验](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ) <br>
:file_folder: [Datasets](docs/DatasetPreparation.md) :arrow_double_down: [Google Drive](https://drive.google.com/drive/folders/1gt5eT293esqY0yr1Anbm36EdnxWW_5oH?usp=sharing) :arrow_double_down: [百度网盘](https://pan.baidu.com/s/1AZDcEAFwwc1OC3KCd7EDnQ) (提取码:basr)<br>
:chart_with_upwards_trend: [Training curves in wandb](https://app.wandb.ai/xintao/basicsr) <br>
:computer: [Commands for training and testing](docs/TrainTest.md) <br>
:zap: [HOWTOs](#zap-howtos)

---

BasicSR (**Basic** **S**uper **R**estoration) is an open source **image and video restoration** toolbox based on PyTorch, such as super-resolution, denoise, deblurring, JPEG artifacts removal, *etc*.<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>
<sub>([HandyView](https://github.com/xinntao/HandyView), [HandyFigure](https://github.com/xinntao/HandyFigure), [HandyCrawler](https://github.com/xinntao/HandyCrawler), [HandyWriting](https://github.com/xinntao/HandyWriting))</sub>

## :sparkles: New Features

- Nov 29, 2020. Add **ESRGAN** and **DFDNet** [colab demo](colab).
- Sep 8, 2020. Add **blind face restoration** inference codes: [DFDNet](https://github.com/csxmli2016/DFDNet).
- Aug 27, 2020. Add **StyleGAN2 training and testing** codes: [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch).

<details>
  <summary>More</summary>
<ul>
  <li> Sep 8, 2020. Add <b>blind face restoration</b> inference codes: <b>DFDNet</b>. <br> <i><font color="#DCDCDC">ECCV20: Blind Face Restoration via Deep Multi-scale Component Dictionaries</font></i> <br> <i><font color="#DCDCDC">Xiaoming Li, Chaofeng Chen, Shangchen Zhou, Xianhui Lin, Wangmeng Zuo and Lei Zhang</font></i> </li>
  <li> Aug 27, 2020. Add <b>StyleGAN2</b> training and testing codes. <br> <i><font color="#DCDCDC">CVPR20: Analyzing and Improving the Image Quality of StyleGAN</font></i> <br> <i><font color="#DCDCDC">Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen and Timo Aila</font></i> </li>
  <li>Aug 19, 2020. A <b>brand-new</b> BasicSR v1.0.0 online.</li>
</ul>
</details>

## :zap: HOWTOs

We provides simple pipelines to train/test/inference models for quick start.
These pipelines/commands cannot cover all the cases and more details are in the following sections.

| GAN |  |  |  | | |
| :--- | :---:        |     :---:      | :--- | :---:        |     :---:      |
| StyleGAN2   | [Train](docs/HOWTOs.md#How-to-train-StyleGAN2) | [Inference](docs/HOWTOs.md#How-to-inference-StyleGAN2) | | | |
| **Face Restoration** |  |  |  | | |
| DFDNet | - | [Inference](docs/HOWTOs.md#How-to-inference-DFDNet) | | | |
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

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Pip install

```bash
pip install basicsr
```

- pip installation does not compile cuda extensions.
- If you want to use cuda extensions, set environment variable `BASICSR_JIT=True`. Note that every time you run the model, it will compile the extensions just time.
  - Example: StyleGAN2 inference colab.

### Git clone and compile

1. Clone repo

    ```bash
    git clone https://github.com/xinntao/BasicSR.git
    ```

1. Install dependent packages

    ```bash
    cd BasicSR
    pip install -r requirements.txt
    ```

1. Install BasicSR

    Please run the following commands in the **BasicSR root path** to install BasicSR:<br>
    (Make sure that your GCC version: gcc >= 5) <br>
    If you do need the cuda extensions: <br>
    &emsp;[*dcn* for EDVR](basicsr/ops)<br>
    &emsp;[*upfirdn2d* and *fused_act* for StyleGAN2](basicsr/ops)<br>
    please add `--cuda_ext` when installing.<br>
    If you use the EDVR and StyleGAN2 model, the above cuda extensions are necessary.

    ```bash
    python setup.py develop --cuda_ext
    ```

    Otherwise, install without compiling cuda extensions

    ```bash
    python setup.py develop
    ```

    You may also want to specify the CUDA paths:

      ```bash
      CUDA_HOME=/usr/local/cuda \
      CUDNN_INCLUDE_DIR=/usr/local/cuda \
      CUDNN_LIB_DIR=/usr/local/cuda \
      python setup.py develop
      ```

Note that BasicSR is only tested in Ubuntu, and may be not suitable for Windows. You may try [Windows WSL with CUDA supports](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) :-) (It is now only available for insider build with Fast ring).

## :hourglass_flowing_sand: TODO List

Please see [project boards](https://github.com/xinntao/BasicSR/projects).

## :turtle: Dataset Preparation

- Please refer to **[DatasetPreparation.md](docs/DatasetPreparation.md)** for more details.
- The descriptions of currently supported datasets (`torch.utils.data.Dataset` classes) are in [Datasets.md](docs/Datasets.md).

## :computer: Train and Test

- **Training and testing commands**: Please see **[TrainTest.md](docs/TrainTest.md)** for the basic usage.
- **Options/Configs**: Please refer to [Config.md](docs/Config.md).
- **Logging**: Please refer to [Logging.md](docs/Logging.md).

## :european_castle: Model Zoo and Baselines

- The descriptions of currently supported models are in [Models.md](docs/Models.md).
- **Pre-trained models and log examples** are available in **[ModelZoo.md](docs/ModelZoo.md)**.
- We also provide **training curves** in [wandb](https://app.wandb.ai/xintao/basicsr):

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="./assets/wandb.jpg" height="280">
</a></p>

## :memo: Codebase Designs and Conventions

Please see [DesignConvention.md](docs/DesignConvention.md) for the designs and conventions of the BasicSR codebase.<br>
The figure below shows the overall framework. More descriptions for each component: <br>
**[Datasets.md](docs/Datasets.md)**&emsp;|&emsp;**[Models.md](docs/Models.md)**&emsp;|&emsp;**[Config.md](Config.md)**&emsp;|&emsp;**[Logging.md](docs/Logging.md)**

![overall_structure](./assets/overall_structure.png)

## :scroll: License and Acknowledgement

This project is released under the Apache 2.0 license.<br>
More details about **license** and **acknowledgement** are in [LICENSE](LICENSE/README.md).

## :earth_asia: Citations

If BasicSR helps your research or work, please consider citing BasicSR.<br>
The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.

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

## :e-mail: Contact

If you have any question, please email `xintao.wang@outlook.com`.
