# HOWTOs

[English](HOWTOs.md) **|** [简体中文](HOWTOs_CN.md)

## 如何训练 StyleGAN2

1. 准备训练数据集: [FFHQ](https://github.com/NVlabs/ffhq-dataset). 更多细节: [DatasetPreparation_CN.md](DatasetPreparation_CN.md#StyleGAN2)
    1. 下载 FFHQ 数据集. 推荐从 [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) 下载 tfrecords 文件.
    1. 从tfrecords 提取到*图片*或者*LMDB*. (需要安装 TensorFlow 来读取 tfrecords).

        > python scripts/data_preparation/extract_images_from_tfrecords.py

1. 修改配置文件 `options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml`
1. 使用分布式训练. 更多训练命令: [TrainTest_CN.md](TrainTest_CN.md)

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml --launcher pytorch

## 如何测试 StyleGAN2

1. 从 **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) 下载预训练模型到 `experiments/pretrained_models` 文件夹.
1. 测试.

    > python inference/inference_stylegan2.py

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

    >  python inference/inference_dfdnet.py --upscale_factor=2 --test_path datasets/TestWhole

6. 结果在 `results/DFDNet` 文件夹.

## How to train SwinIR (SR)

We take the classical SR X4 with DIV2K for example.

1. Prepare the training dataset: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/). More details are in [DatasetPreparation.md](DatasetPreparation.md#image-super-resolution)
1. Prepare the validation dataset: Set5. You can download with [this guidance](DatasetPreparation.md#common-image-sr-datasets)
1. Modify the config file in [`options/train/SwinIR/train_SwinIR_SRx4_scratch.yml`](../options/train/SwinIR/train_SwinIR_SRx4_scratch.yml) accordingly.
1. Train with distributed training. More training commands are in [TrainTest.md](TrainTest.md).

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4331 basicsr/train.py -opt options/train/SwinIR/train_SwinIR_SRx4_scratch.yml --launcher pytorch  --auto_resume

Note that:

1. Different from the original setting in the paper where the X4 model is finetuned from the X2 model, we directly train it from scratch.
1. We also use `EMA (Exponential Moving Average)`. Note that all model trainings in BasicSR supports EMA.
1. In the **250K iteration** of training X4 model, it can achieve comparable performance to the official model.

|  ClassicalSR DIV2KX4 | PSNR (RGB) | PSNR (Y) | SSIM (RGB)  | SSIM (Y) |
| :--- | :---:        |     :---:      | :---: | :---:        |
|  Official  | 30.803 | 32.728 | 0.8738|0.9028 |
|  Reproduce |30.832  | 32.756 | 0.8739| 0.9025 |

## How to inference SwinIR (SR)

1. Download pre-trained models from the [**official SwinIR repo**](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) to the `experiments/pretrained_models/SwinIR` folder.
1. Inference.

    > python inference/inference_swinir.py --input datasets/Set5/LRbicx4 --patch_size 48 --model_path experiments/pretrained_models/SwinIR/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth --output results/SwinIR_SRX4_DIV2K/Set5

1. The results are in the `results/SwinIR_SRX4_DIV2K/Set5` folder.
1. You may want to calculate the PSNR/SSIM values.

    > python scripts/metrics/calculate_psnr_ssim.py --gt datasets/Set5/GTmod12/ --restored results/SwinIR_SRX4_DIV2K/Set5 --crop_border 4

    or test with the Y channel with the `--test_y_channel` argument.

    > python scripts/metrics/calculate_psnr_ssim.py --gt datasets/Set5/GTmod12/ --restored results/SwinIR_SRX4_DIV2K/Set5 --crop_border 4  --test_y_channel
