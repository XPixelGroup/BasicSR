# HOWTOs

[English](HOWTOs.md) **|** [简体中文](HOWTOs_CN.md)

## How to train StyleGAN2

1. Prepare training dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset). More details are in [DatasetPreparation.md](DatasetPreparation.md#StyleGAN2)
    1. Download FFHQ dataset. Recommend to download the tfrecords files from [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).
    1. Extract tfrecords to images or LMDBs (TensorFlow is required to read tfrecords):

        > python scripts/extract_images_from_tfrecords.py

1. Modify the config file in `options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml`
1. Train with distributed training. More training commands are in [TrainTest.md](TrainTest.md).

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ_800k.yml --launcher pytorch

## How to test StyleGAN2

1. Download pre-trained models from **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) to the `experiments/pretrained_models` folder.
1. Test.

    > python tests/test_stylegan2.py

1. The results are in the `samples` folder.

## How to test DFDNet

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

    >  python tests/test_face_dfdnet.py --upscale_factor=2 --test_path datasets/TestWhole

6. The results are in the `results/DFDNet` folder.
