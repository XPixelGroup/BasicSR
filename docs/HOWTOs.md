# HOWTOs

[English](HOWTOs.md) | [简体中文](HOWTOs_CN.md)

## How to train StyleGAN2

1. Prepare training dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset). More details are in [DatasetPreparation.md](DatasetPreparation.md#StyleGAN2)
    1. Download FFHQ dataset. Recommend to download the tfrecords files from [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).
    1. Extract tfrecords to images or LMDBs (TensorFlow is required to read tfrecords):

        > python scripts/extract_images_from_tfrecords.py

1. Modify the config file in `options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml`
1. Train with distributed training. More training commands are in [TrainTest.md](TrainTest.md).

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ_800k.yml --launcher pytorch

## How to test StyleGAN2

1. Test:
    1. Download pre-trained models from [ModelZoo](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing) to the `experiments/pretrained_models` folder.
    1. Test.

        > python tests/test_stylegan2.py

    1. The results are in the `samples` folder.
