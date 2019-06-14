## BasicSR [[EDVR]](https://github.com/xinntao/EDVR) [[DNI]](https://xinntao.github.io/projects/DNI) [[ESRGAN]](https://github.com/xinntao/ESRGAN) [[SFTGAN]](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/)

:triangular_flag_on_post: We have updated the BasicSR toolbox (v0.1).<br/>
Almost all the files have updates, including:
- [x] Support PyTorch 1.1 and distributed training
- [x] Simplify network structures
- [x] Update the dataset format
- [x] Use *yaml* for configurations
- [x] ...

If you find compatibility issues, please see whether these files are in the [To-be-updated list](https://github.com/xinntao/BasicSR/blob/master/updateTODO.txt).<br/>
If you want to use the old version, please find it in the [releases](https://github.com/xinntao/BasicSR/releases) with `tag v0.0`.

---
Check out our new work on:<br/>
1. **Video Super-Resolution**: [`EDVR: Video Restoration with Enhanced Deformable Convolutional Networks`](https://xinntao.github.io/projects/EDVR), which has won all four tracks in NTIRE 2019 Challenges on Video Restoration and Enhancement (CVPR19 Workshops).
2. **DNI (CVPR19)**: [`Deep Network Interpolation for Continuous Imagery Effect Transition`](https://xinntao.github.io/projects/DNI)
---

<p align="center">
  <img height="400" src="https://github.com/xinntao/ESRGAN/blob/master/figures/baboon.jpg">
</p>

### Updates
[2019-06-13] Update to a new version.<br/>

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard: 
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`
  
## Dataset Preparation
We use datasets in LDMB format for faster IO speed. Please refer to [wiki](https://github.com/xinntao/BasicSR/wiki/Prepare-datasets-in-LMDB-format) for more details.

## Get Started
Please see [wiki](https://github.com/xinntao/BasicSR/wiki/Training-and-Testing) for the basic usage, *i.e.,* training and testing.

## Model Zoo and Baselines
Results and pre-trained models are available in the [wiki-Model zoo](https://github.com/xinntao/BasicSR/wiki/Model-Zoo).

## Contributing
We appreciate all contributions. Please refer to [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/CONTRIBUTING.md) for contributing guideline.

**Python code style**<br/>
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style. We use [flake8](http://flake8.pycqa.org/en/latest/) as the linter and [yapf](https://github.com/google/yapf) as the formatter. Please upgrade to the latest yapf (>=0.27.0) and refer to the [yapf configuration](https://github.com/xinntao/BasicSR/blob/master/.style.yapf) and [flake8 configuration](https://github.com/xinntao/BasicSR/blob/master/.flake8).

> Before you create a PR, make sure that your code lints and is formatted by yapf.

## Citation

    @InProceedings{wang2018esrgan,
        author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
        title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
        booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
        month = {September},
        year = {2018}
    }
    @InProceedings{wang2018sftgan,
        author = {Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
        title = {Recovering realistic texture in image super-resolution by deep spatial feature transform},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2018}
    }

