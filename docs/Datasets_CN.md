# 数据处理

[English](Datasets.md) **|** [简体中文](Datasets_CN.md)

## 支持的数据处理

| 类         | 任务    |训练/测试 | 描述       |
| :------------- | :----------:| :----------:    | :----------:   |
| [PairedImageDataset](../basicsr/data/paired_image_dataset.py) | 图像超分 | 训练|支持读取成对的训练数据 |
| [SingleImageDataset](../basicsr/data/single_image_dataset.py) | 图像超分 | 测试|只读取low quality的图像, 用在没有Ground-Truth的测试中 |
| [REDSDataset](../basicsr/data/reds_dataset.py) | 视频超分 | 训练|REDS的训练数据集 |
| [Vimeo90KDataset](../basicsr/data/vimeo90k_dataset.py) | 视频超分 |训练| Vimeo90K的训练数据集|
| [VideoTestDataset](../basicsr/data/video_test_dataset.py) | 视频超分 | 测试|基础的视频超分测试集, 支持Vid4, REDS测试集|
| [VideoTestVimeo90KDataset](../basicsr/data/video_test_dataset.py) | 视频超分 |测试| 继承`VideoTestDataset`, Vimeo90K的测试数据集|
| [VideoTestDUFDataset](../basicsr/data/video_test_dataset.py) | 视频超分 |测试| 继承`VideoTestDataset`, 方法DUF的测试数据集, 支持Vid4|
| [FFHQDataset](../basicsr/data/ffhq_dataset.py) | 人脸生成 |训练| FFHQ的训练数据集|

1. 共用的变换和函数分别在 [transforms.py](../basicsr/data/transforms.py) 和 [util.py](../basicsr/data/util.py) 中
