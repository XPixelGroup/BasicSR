# Metrics

[English](README.md) **|** [简体中文](README_CN.md)

- [约定](#约定)
- [PSNR 和 SSIM](#psnr-和-ssim)

## 约定

因为不同的输入类型会导致结果的不同，因此我们对输入做如下约定:

- Numpy 类型 (一般是 cv2 的结果)
  - UINT8: BGR, [0, 255], (h, w, c)
  - float: BGR, [0, 1], (h, w, c). 一般作为中间结果
- Tensor 类型
  - float: RGB, [0, 1], (n, c, h, w)

其他约定:

- 以 `_pt` 结尾的是 PyTorch 结果
- PyTorch version 支持 batch 计算
- 颜色转换在 float32 上做；metric计算在 float64 上做

## PSNR 和 SSIM

PSNR 和 SSIM 的结果趋势是一致的，即一般 PSNR 高，则 SSIM 也高。
在实现上, PSNR 的各种实现都很一致。SSIM 有各种各样的实现，我们这里和 MATLAB 最原始版本保持 (参考 [NTIRE17比赛](https://competitions.codalab.org/competitions/16306#participate) 的 [evaluation代码](https://competitions.codalab.org/my/datasets/download/ebe960d8-0ec8-4846-a1a2-7c4a586a7378))

下面列了各个实现的结果比对.
总结：PyTorch 实现和 MATLAB 实现基本一致，在 GPU 运行上会有稍许差异

- PSNR 比对

|Image | Color Space | MATLAB | Numpy | PyTorch CPU | PyTorch GPU  |
|:---| :---: | :---:  | :---:      |     :---:      | :---: |
|baboon| RGB |  20.419710  | 20.419710 | 20.419710 |20.419710 |
|baboon| Y | - |22.441898 | 22.441899 |  22.444916|
|comic | RGB | 20.239912 | 20.239912 | 20.239912 | 20.239912 |
|comic | Y | - | 21.720398 | 21.720398  | 21.721663|

- SSIM 比对

|Image | Color Space | MATLAB | Numpy | PyTorch CPU | PyTorch GPU  |
|:---| :---: | :---:  | :---:      |     :---:      | :---: |
|baboon| RGB |  0.391853  | 0.391853 | 0.391853|0.391853 |
|baboon| Y | - |0.453097| 0.453097 |  0.453171|
|comic | RGB | 0.567738 | 0.567738 | 0.567738 | 0.567738|
|comic | Y | - | 0.585511 | 0.585511 | 0.585522 |
