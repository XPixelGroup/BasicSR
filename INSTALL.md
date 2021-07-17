# Installation

## Contents

- [Requirements](#requirements)
    - [BASICSR_EXT and BASICSR_JIT enviroment variables](#basicsr_ext-and-basicsr_jit-environment-variables)
- [Installation Options](#installation-options)
  - [Install from PyPI](#install-from-pypi)
  - [Install from a local clone](#Install-from-a-local-clone)
- [FAQ](#faq)

## Requirements

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Linux (We have not tested on Windows)

### BASICSR_EXT and BASICSR_JIT environment variables

If you want to use deformable convolution or StyleGAN customized operators, you also need to 1) compile the PyTorch C++ extensions ahead OR 2) load the PyTorch C++ extensions just-in-time (JIT).<br>
You may choose one of the options according to your needs.

| Option | Pros| Cons | Cases |
| :--- | :---        |     :---      | :--- |
| Compile the PyTorch C++ extensions ahead   | Quickly load the compiled extensions | May have more stringent requirements for the environment, and you may encounter annoying issues | If you need to train/inference those models for many times, it will save your time|
| Load the PyTorch C++ extensions just-in-time (JIT) | Have less requirements for running | Each time you run the model, it will load extensions again, which may takes several minutes  | If you just want to do some inferences, it will reduce the issues you may encounter|

For those who need to compile the PyTorch C++ extensions ahead, remember:

- Make sure that your GCC version: gcc >= 5

If you do not need these operators, just skip it and there is no need to set `BASICSR_EXT` or `BASICSR_JIT` environment variables.


## Installation Options

### Install from PyPI

```bash
pip install basicsr
```

- If you want to compile cuda extensions when installing, please set up the environment variable `BASICSR_EXT=True`.

  ```bash
  BASICSR_EXT=True pip install basicsr
  ```

- If you want to use cuda extensions during running, set environment variable `BASICSR_JIT=True`. Note that every time you run the model, it will compile the extensions just time.
  - Example: StyleGAN2 inference colab.

### Install from a local clone

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
    please set up the environment variable `BASICSR_EXT=True` when installing.<br>
    If you use the EDVR and StyleGAN2 model, the above cuda extensions are necessary.

    ```bash
    BASICSR_EXT=True python setup.py develop
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
