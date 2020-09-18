# 模型

[English](Models.md) **|** [简体中文](Models_CN.md)

#### 目录

1. [支持的模型](#支持的模型)
1. [继承关系](#继承关系)

## 支持的模型

| 类         | 描述    |支持的算法 |
| :------------- | :----------:| :----------:    |
| [BaseModel](../basicsr/models/base_model.py) | 抽象基类, 定义了共有的函数||
| [SRModel](../basicsr/models/sr_model.py) | 基础图像超分类 | SRCNN, EDSR, SRResNet, RCAN, RRDBNet, etc |
| [SRGANModel](../basicsr/models/srgan_model.py) | SRGAN图像超分类 | SRGAN |
| [ESRGANModel](../basicsr/models/esrgan_model.py) | ESRGAN图像超分类 |ESRGAN|
| [VideoBaseModel](../basicsr/models/video_base_model.py) | 基础视频超分类 | |
| [EDVRModel](../basicsr/models/edvr_model.py) | EDVR视频超分类 |EDVR|
| [StyleGAN2Model](../basicsr/models/stylegan2_model.py) | StyleGAN2图像生成类 |StyleGAN2|

## 继承关系

为增加模型间的复用, 我们大量使用了继承, 以下为各个模型之间的继承关系:

```txt
BaseModel
├── SRModel
│   ├── SRGANModel
│   │   ├── ESRGANModel
│   ├── VideoBaseModel
│       ├── EDVRModel
├── StyleGAN2Model
```
