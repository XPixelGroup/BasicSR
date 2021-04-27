# HiFaceGAN Tutorial

[English](HiFaceGAN.md) **|** [简体中文](HiFaceGAN_CN.md) | [原工程地址](https://github.com/Lotayou/Face-Renovation)

## How to run HiFaceGAN inference script
1. Create testing data. Current models only support 512 resolution images (use cv2.INTER_CUBIC for rescaling). For academic benchmarks, we use aligned data format of [LQ;HQ] concatenated along the height dimension, so each aligned pair is of shape 1024*512*3(HWC). If you wish to run inference on in-the-wild images, no GT is required.

Save all images into a single folder , denote the path as `xxx`.
2. Download pretrained checkpoints and denote the unzipped folder as `yyy`:

[BaiduNetDisk](https://pan.baidu.com/s/1lp-mj5LaTfNxAxrn4QOcSA) Code：rwzh 

P.S. I cannot access google drive for now, request assistance from xintao.

3. Modify the following parameters in the configuration file `/backup/lingbo/projects/BasicSR/basicsr/models/archs/hifacegan_options.py`:

    ```
        dataroot = xxx
        checkpoints_dir = yyy
        name = <subtask type>（default: generic -> works best for in-the-wild images）
    ```

4. Run the script and check the result at `results/HiFaceGAN`. Some testing results under generic mode are already provided.
    >  python inference_hifacegan.py

## Version History

#### v0.1: 
- Merge HiFaceGAN network architecture and configuration file. Provide a simple testing script, some in-the-wild samples and the corresponding inference results.

## TODO List:
- Solve import bug: currently `import basicsr` returns `ModuleNotFoundError` when trying to executing script within subfolders. `python inference/inference_hifacegan.py` does not work
- Migrate training codes and data preparation scripts.
- Add support for lmdb format dataset.
- Add support for distributed dataparallel (ETA: 2021.10 or later)
