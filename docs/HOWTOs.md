# HOWTOs

[English](HOWTOs.md) **|** [简体中文](HOWTOs_CN.md)

## How to train StyleGAN2

1. Prepare training dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset). More details are in [DatasetPreparation.md](DatasetPreparation.md#StyleGAN2)
    1. Download FFHQ dataset. Recommend to download the tfrecords files from [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).
    1. Extract tfrecords to images or LMDBs (TensorFlow is required to read tfrecords):

        > python scripts/data_preparation/extract_images_from_tfrecords.py

1. Modify the config file in `options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml`
1. Train with distributed training. More training commands are in [TrainTest.md](TrainTest.md).

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ_800k.yml --launcher pytorch

## How to inference StyleGAN2

1. Download pre-trained models from **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) to the `experiments/pretrained_models` folder.
1. Test.

    > python inference/inference_stylegan2.py

1. The results are in the `samples` folder.

## How to inference DFDNet

1. Install [dlib](http://dlib.net/), because DFDNet uses dlib to do face recognition and landmark detection. [Installation reference](https://github.com/davisking/dlib).
    1. Clone dlib repo: `git clone git@github.com:davisking/dlib.git`
    1. `cd dlib`
    1. Install: `python setup.py install`
2. Download the dlib pretrained models from **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) to the `experiments/pretrained_models/dlib` folder.<br>
    You can download by run the following command OR manually download the pretrained models.

    > python scripts/download_pretrained_models.py dlib

3. Download pretrained DFDNet models, dictionary and face template from **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) to the `experiments/pretrained_models/DFDNet` folder.<br>
    You can download by run the the following command OR manually download the pretrained models.

    > python scripts/download_pretrained_models.py DFDNet

4. Prepare the testing dataset in the `datasets`, for example, we put images in the `datasets/TestWhole` folder.
5. Test.

    >  python inference/inference_dfdnet.py --upscale_factor=2 --test_path datasets/TestWhole

6. The results are in the `results/DFDNet` folder.

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
