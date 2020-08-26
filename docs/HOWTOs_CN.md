# HOWTOs

[English](HOWTOs.md) | [简体中文](HOWTOs_CN.md)

## 如何训练 StyleGAN2

1. 准备训练数据集: [FFHQ](https://github.com/NVlabs/ffhq-dataset). 更多细节: [DatasetPreparation_CN.md](DatasetPreparation_CN.md#StyleGAN2)
    1. 下载 FFHQ 数据集. 推荐从 [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) 下载 tfrecords 文件.
    1. 从tfrecords 提取到*图片*或者*LMDB*. (需要安装 TensorFlow 来读取 tfrecords).

        > python scripts/extract_images_from_tfrecords.py

1. 修改配置文件 `options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml`
1. 使用分布式训练. 更多训练命令: [TrainTest_CN.md](TrainTest_CN.md)

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml --launcher pytorch

## 如何测试 StyleGAN2

1. 测试:
    1. 从 [ModelZoo](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing) 下载预训练模型到 `experiments/pretrained_models` 文件夹.
    1. 测试.

        > python tests/test_stylegan2.py

    1. 结果在 `samples` 文件夹
