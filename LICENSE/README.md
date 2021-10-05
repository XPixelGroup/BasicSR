# License and Acknowledgement

This BasicSR project is released under the Apache 2.0 license.

- StyleGAN2
  - The codes are modified from the repository [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch). Many thanks to the author - [Kim Seonghyeon](https://rosinality.github.io/)  :blush: for translating from the official TensorFlow codes to PyTorch ones. Here is the [license](LICENSE-stylegan2-pytorch) of stylegan2-pytorch.
  - The official repository is https://github.com/NVlabs/stylegan2, and here is the [NVIDIA license](./LICENSE-NVIDIA).
- DFDNet
  - The codes are largely modified from the repository [DFDNet](https://github.com/csxmli2016/DFDNet). Their license is [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- DiffJPEG
  - Modified from https://github.com/mlomnitz/DiffJPEG.
- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/)
  - We use the implementation of `DropPath` and `trunc_normal_` from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/). The LICENSE is included as [LICENSE_pytorch-image-models](LICENSE/LICENSE_pytorch-image-models).
- [SwinIR](https://github.com/JingyunLiang/SwinIR)
  - The arch implementation of SwinIR is from [SwinIR](https://github.com/JingyunLiang/SwinIR). The LICENSE is included as [LICENSE_SwinIR](LICENSE/LICENSE_SwinIR).
- [ECBSR](https://github.com/xindongzhang/ECBSR)
  - The arch implementation of ECBSR is from [ECBSR](https://github.com/xindongzhang/ECBSR). The LICENSE of ECBSR is [Apache License 2.0](https://github.com/xindongzhang/ECBSR/blob/main/LICENSE)

## References

1. NIQE metric: the codes are translated from the [official MATLAB codes](http://live.ece.utexas.edu/research/quality/niqe_release.zip)

    > A. Mittal, R. Soundararajan and A. C. Bovik, "Making a Completely Blind Image Quality Analyzer", IEEE Signal Processing Letters, 2012.

1. FID metric: the codes are modified from [pytorch-fid](https://github.com/mseitzer/pytorch-fid) and [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).
