# Installation

## Contents

- [Requirements](#requirements)
- [BASICSR_EXT and BASICSR_JIT environment variables](#basicsr_ext-and-basicsr_jit-environment-variables)
- [Installation Options](#installation-options)
  - [Install from PyPI](#install-from-pypi)
  - [Install from a local clone](#Install-from-a-local-clone)
- [FAQ](#faq)

## Requirements

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Linux (We have not tested on Windows)

## BASICSR_EXT and BASICSR_JIT Environment Variables

If you want to use PyTorch C++ extensions:<br>
&emsp;deformable convolution: [*dcn* for EDVR](basicsr/ops) (For torchvision>=0.9.0, we use the official `torchvision.ops.deform_conv2d` instead)<br>
&emsp;StyleGAN customized operators: [*upfirdn2d* and *fused_act* for StyleGAN2](basicsr/ops)<br>
you also need to:

1. **compile** the PyTorch C++ extensions during installation
2. OR **load** the PyTorch C++ extensions just-in-time (JIT)

You may choose one of the options according to your needs.

| Option | Pros| Cons | Cases | Env Variable|
| :--- | :---        |     :---      | :--- |:--- |
| **Compile** PyTorch C++ extensions during installation   | **Quickly load** the compiled extensions during running | May have more stringent requirements for the environment, and you may encounter annoying issues | If you need to train/inference those models for many times, it will save your time| Set `BASICSR_EXT=True` during **installation**|
| **Load** PyTorch C++ extensions just-in-time (JIT) | Have less requirements, may have less issues | Each time you run the model, it will takes several minutes to load extensions again  | If you just want to do simple inferences, it is more convenient| Set  `BASICSR_JIT=True` during **running** (not **installation**) |

For those who need to compile the PyTorch C++ extensions during installation, remember:

- Make sure that your gcc and g++ version: gcc & g++ >= 5

Note that:

- The `BASICSR_JIT` has higher priority, that is, even you have successfully compiled PyTorch C++ extensions during installation, it will still load the extensions just-in-time if you set `BASICSR_JIT=True` in your running commands.
- :x: Do not set `BASICSR_JIT` during installation. Installation commands are in [Installation Options](#installation-options).
- :heavy_check_mark: If you want to load PyTorch C++ extensions just-in-time (JIT), just set `BASICSR_JIT=True` before your  **running** commands. For example, `BASICSR_JIT=True python inference/inference_stylegan2.py`.

If you do not need those PyTorch C++ extensions, just skip it. There is no need to set `BASICSR_EXT` or `BASICSR_JIT` environment variables.

## Installation Options

There are two options to install BASICSR, according to your needs.

- If you just want to use BASICSR as a **package** (just like [GFPGAN](https://github.com/TencentARC/GFPGAN) and []()), it is recommended to install from PyPI.
- If you want to **investigate** the details of BASICSR OR **develop** it OR **modify** it to fulfill your needs, it is better to install from a local clone.

### Install from PyPI

- If you do not need C++ extensions (more details are [here](#basicsr_ext-and-basicsr_jit-environment-variables)):

  ```bash
  pip install basicsr
  ```

- If you want to use C++ extensions in **JIT mode** without compiling them during installatoin (more details are [here](#basicsr_ext-and-basicsr_jit-environment-variables)):

  ```bash
  pip install basicsr
  ```

- If you want to **compile C++ extensions during installation**, please set the environment variable `BASICSR_EXT=True`:

  ```bash
  BASICSR_EXT=True pip install basicsr
  ```

  The compilation may fail without any error prints. If you encounter running errors, such as `ImportError: cannot import name 'deform_conv_ext' | 'fused_act_ext' | 'upfirdn2d_ext'`, you may check the compilation process by re-installation. The following command will print detailed log:

  ```bash
  BASICSR_EXT=True pip install basicsr -vvv
  ```

  You may also want to specify the CUDA paths:

  ```bash
  CUDA_HOME=/usr/local/cuda \
  CUDNN_INCLUDE_DIR=/usr/local/cuda \
  CUDNN_LIB_DIR=/usr/local/cuda \
  BASICSR_EXT=True pip install basicsr
  ```

### Install from a local clone

1. Clone the repo

    ```bash
    git clone https://github.com/xinntao/BasicSR.git
    ```

1. Install dependent packages

    ```bash
    cd BasicSR
    pip install -r requirements.txt
    ```

1. Install BasicSR<br>
    Please run the following commands in the **BasicSR root path** to install BasicSR:<br>

    -  If you do not need C++ extensions (more details are [here](#basicsr_ext-and-basicsr_jit-environment-variables)):

        ```bash
        python setup.py develop
        ```

    - If you want to use C++ extensions in **JIT mode** without compiling them during installatoin (more details are [here](#basicsr_ext-and-basicsr_jit-environment-variables)):

        ```bash
        python setup.py develop
        ```

    - If you want to **compile C++ extensions during installation**, please set the environment variable `BASICSR_EXT=True`:

        ```bash
        BASICSR_EXT=True python setup.py develop
        ```

    You may also want to specify the CUDA paths:

    ```bash
    CUDA_HOME=/usr/local/cuda \
    CUDNN_INCLUDE_DIR=/usr/local/cuda \
    CUDNN_LIB_DIR=/usr/local/cuda \
    BASICSR_EXT=True python setup.py develop
    ```

## FAQ
