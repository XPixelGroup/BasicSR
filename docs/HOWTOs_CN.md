# HOWTOs

[English](HOWTOs.md) | [简体中文](HOWTOs_CN.md)

## 如何训练 StyleGAN2

1. 准备训练数据集: [FFHQ](https://github.com/NVlabs/ffhq-dataset). 更多细节: [DatasetPreparation_CN.md](DatasetPreparation_CN.md#StyleGAN2)
    1. 下载 FFHQ 数据集. 推荐从 [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) 下载 tfrecords 文件.
    1. 从tfrecords 提取到*图片*或者*LMDB*. (需要安装 TensorFlow 来读取 tfrecords).

        > python scripts/extract_images_from_tfrecords.py

1. Modify the config file in `options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ_800k.yml`
1. Train with distributed training:

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ_800k.yml --launcher pytorch

## How to test StyleGAN2

1. Test:

    > python tests/test_stylegan2.py
