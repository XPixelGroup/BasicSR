# HOWTOs

[English](HOWTOs.md) **|** [简体中文](HOWTOs_CN.md)

## 如何训练 StyleGAN2

1. 准备训练数据集: [FFHQ](https://github.com/NVlabs/ffhq-dataset). 更多细节: [DatasetPreparation_CN.md](DatasetPreparation_CN.md#StyleGAN2)
    1. 下载 FFHQ 数据集. 推荐从 [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) 下载 tfrecords 文件.
    1. 从tfrecords 提取到*图片*或者*LMDB*. (需要安装 TensorFlow 来读取 tfrecords).

        > python scripts/extract_images_from_tfrecords.py

1. 修改配置文件 `options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml`
1. 使用分布式训练. 更多训练命令: [TrainTest_CN.md](TrainTest_CN.md)

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml --launcher pytorch

## 如何测试 StyleGAN2

1. 从 **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) 下载预训练模型到 `experiments/pretrained_models` 文件夹.
1. 测试.

    > python tests/test_stylegan2.py

1. 结果在 `samples` 文件夹

## 如何测试 DFDNet

1. 安装 [dlib](http://dlib.net/). 因为 DFDNet 使用 dlib 做人脸检测和关键点检测. [安装参考](https://github.com/davisking/dlib).
    1. 克隆 dlib repo: `git clone git@github.com:davisking/dlib.git`
    1. `cd dlib`
    1. 安装: `python setup.py install`
2. 从 **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) 下载预训练的 dlib 模型到 `experiments/pretrained_models/dlib` 文件夹.<br>
    你可以通过运行下面的命令下载 或 手动下载.

    > python scripts/download_pretrained_models.py dlib

3. 从 **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) 下载 DFDNet 模型, 字典和人脸关键点模板到 `experiments/pretrained_models/DFDNet` 文件夹.<br>
     你可以通过运行下面的命令下载 或 手动下载.

    > python scripts/download_pretrained_models.py DFDNet

4. 准备测试图片到 `datasets`, 比如说我们把测试图片放在 `datasets/TestWhole` 文件夹.
5. 测试.

    >  python tests/test_face_dfdnet.py --upscale_factor=2 --test_path datasets/TestWhole

6. 结果在 `results/DFDNet` 文件夹.
