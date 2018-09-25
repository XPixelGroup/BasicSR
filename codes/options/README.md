# Configurations
- Use **json** files to configure options.
- Convert the json file to python dict.
- Support `//` comments and use `null` for `None`.

## Table
Click for detailed explanations for each json file.

1. [train_sr.json](#train_sr_json)
1. [train_esrgan.json](#train_esrgan_json) (also for training srgan)
1. [train_sftgan.json](#train_sftgan_json)

## train_sr_json
```c++
{
  "name": "debug_001_RRDB_PSNR_x4_DIV2K" //  leading 'debug_' enters the 'debug' mode. Please remove it during formal training
  , "use_tb_logger": true // use tensorboard_logger, ref: `https://github.com/xinntao/BasicSR/tree/master/codes/utils`
  , "model":"sr" // model type, ref: `https://github.com/xinntao/BasicSR/blob/master/codes/models/__init__.py`
  , "scale": 4 // scale factor for SR
  , "gpu_ids": [0] // specify GPUs, actually it sets the `CUDA_VISIBLE_DEVICES`

  , "datasets": { // configure the training and validation datasets
    "train": { // training dataset configurations
      "name": "DIV2K" // dataset name
      , "mode": "LRHR" // dataset mode, ref: `https://github.com/xinntao/BasicSR/blob/master/codes/data/__init__.py`
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb" // HR data root
      , "dataroot_LR": "/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub_bicLRx4.lmdb" // LR data root
      , "subset_file": null // use a subset of an image folder
      , "use_shuffle": true // shuffle the dataset
      , "n_workers": 8 // number of data load workers
      , "batch_size": 16
      , "HR_size": 128 // 128 | 192, cropped HR patch size
      , "use_flip": true // whether use horizontal and vertical flips
      , "use_rot": true // whether use rotations: 90, 190, 270 degrees
    }
    , "val": { // validation dataset configurations
      "name": "val_set5"
      , "mode": "LRHR"
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/val_set5/Set5"
      , "dataroot_LR": "/mnt/SSD/xtwang/BasicSR_datasets/val_set5/Set5_bicLRx4"
    }
  }

  , "path": {
    "root": "/home/xtwang/Projects/BasicSR" // root path
    , "pretrain_model_G": null // path of the pretrained model
  }

  , "network_G": { // configurations for the network G
    "which_model_G": "RRDB_net" // RRDB_net | sr_resnet, network structures, ref: `https://github.com/xinntao/BasicSR/blob/master/codes/models/networks.py`
    , "norm_type": null // null | "batch", norm type 
    , "mode": "CNA" // Convolution mode: CNA for Conv-Norm_Activation
    , "nf": 64 // number of features for each layer
    , "nb": 23 // number of blocks
    , "in_nc": 3 // input channels
    , "out_nc": 3 // output channels
    , "gc": 32 // grouwing channels, for Dense Block
    , "group": 1 // convolution group, for ResNeXt Block
  }

  , "train": { // training strategies
    "lr_G": 2e-4 // initialized learning rate
    , "lr_scheme": "MultiStepLR" // learning rate decay scheme
    , "lr_steps": [200000, 400000, 600000, 800000] // at which steps, decay the learining rate
    , "lr_gamma": 0.5 

    , "pixel_criterion": "l1" // "l1" | "l2", criterion
    , "pixel_weight": 1.0
    , "val_freq": 5e3 // validation frequency

    , "manual_seed": 0
    , "niter": 1e6 // total training iteration
  }

  , "logger": { // logger configurations
    "print_freq": 200
    , "save_checkpoint_freq": 5e3
  }
}
```
## train_esrgan_json
```c++
{
  "name": "debug_002_RRDB_ESRGAN_x4_DIV2K" // leading 'debug_' enters the 'debug' mode. Please remove it during formal training
  , "use_tb_logger": true // use tensorboard_logger, ref: `https://github.com/xinntao/BasicSR/tree/master/codes/utils`
  , "model":"srragan" // "srgan" |"srragan", model type, ref: `https://github.com/xinntao/BasicSR/blob/master/codes/models/__init__.py`
  , "scale": 4 // scale factor for SR
  , "gpu_ids": [0] // specify GPUs, actually it sets the `CUDA_VISIBLE_DEVICES`

  , "datasets": { // configure the training and validation datasets
    "train": { // training dataset configurations
      "name": "DIV2K" // dataset name
      , "mode": "LRHR" // dataset mode, ref: `https://github.com/xinntao/BasicSR/blob/master/codes/data/__init__.py`
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb" // HR data root
      , "dataroot_LR": "/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub_bicLRx4.lmdb" // LR data root
      , "subset_file": null // use a subset of an image folder
      , "use_shuffle": true // shuffle the dataset
      , "n_workers": 8 // number of data load workers
      , "batch_size": 16
      , "HR_size": 128 // 128 | 192, cropped HR patch size
      , "use_flip": true // whether use horizontal and vertical flips
      , "use_rot": true // whether use rotations: 90, 190, 270 degrees
    }
    , "val": { // validation dataset configurations
      "name": "val_set14_part"
      , "mode": "LRHR"
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/val_set14_part/Set14"
      , "dataroot_LR": "/mnt/SSD/xtwang/BasicSR_datasets/val_set14_part/Set14_bicLRx4"
    }
  }

  , "path": {
    "root": "/home/xtwang/Projects/BasicSR" // root path
    , "pretrain_model_G": "../experiments/pretrained_models/RRDB_PSNR_x4.pth" // path of the pretrained model
  }

  , "network_G": { // configurations for the network G
    "which_model_G": "RRDB_net" // RRDB_net | sr_resnet, network structures, ref: `https://github.com/xinntao/BasicSR/blob/master/codes/models/networks.py`
    , "norm_type": null // null | "batch", norm type 
    , "mode": "CNA" // Convolution mode: CNA for Conv-Norm_Activation
    , "nf": 64 // number of features for each layer
    , "nb": 23 // number of blocks
    , "in_nc": 3 // input channels
    , "out_nc": 3 // output channels
    , "gc": 32 // grouwing channels, for Dense Block
    , "group": 1 // convolution group, for ResNeXt Block
  }
  , "network_D": {// configurations for the network D
    "which_model_D": "discriminator_vgg_128" // network structures, ref: `https://github.com/xinntao/BasicSR/blob/master/codes/models/networks.py`
    , "norm_type": "batch"
    , "act_type": "leakyrelu"
    , "mode": "CNA"
    , "nf": 64
    , "in_nc": 3
  }

  , "train": { // training strategies
    "lr_G": 1e-4 // initialized learning rate for G
    , "weight_decay_G": 0
    , "beta1_G": 0.9
    , "lr_D": 1e-4 // initialized learning rate for D
    , "weight_decay_D": 0
    , "beta1_D": 0.9 
    , "lr_scheme": "MultiStepLR" // learning rate decay scheme
    , "lr_steps": [50000, 100000, 200000, 300000] // at which steps, decay the learining rate
    , "lr_gamma": 0.5

    , "pixel_criterion": "l1" // "l1" | "l2", pixel criterion
    , "pixel_weight": 1e-2
    , "feature_criterion": "l1" // perceptual criterion (VGG loss)
    , "feature_weight": 1
    , "gan_type": "vanilla" // GAN type
    , "gan_weight": 5e-3

    //for wgan-gp
    // , "D_update_ratio": 1
    // , "D_init_iters": 0
    // , "gp_weigth": 10

    , "manual_seed": 0
    , "niter": 5e5 // total training iteration
    , "val_freq": 5e3 // validation frequency
  }

  , "logger": { // logger configurations
    "print_freq": 200
    , "save_checkpoint_freq": 5e3
  }
}
```
## train_sftgan_json

```c++
{
  "name": "debug_003_SFTGANx4_OST" // leading 'debug_' enters the 'debug' mode. Please remove it during formal training
  , "use_tb_logger": false // use tensorboard_logger, ref: `https://github.com/xinntao/BasicSR/tree/master/codes/utils`
  , "model": "sftgan" // model type, ref: `https://github.com/xinntao/BasicSR/blob/master/codes/models/__init__.py`
  , "scale": 4 // scale factor for SR
  , "gpu_ids": [0] // specify GPUs, actually it sets the `CUDA_VISIBLE_DEVICES`

  , "datasets": { // configure the training and validation datasets
    "train": { // training dataset configurations
      "name": "OST" // dataset name
      , "mode": "LRHRseg_bg" // dataset mode, ref: `https://github.com/xinntao/BasicSR/blob/master/codes/data/__init__.py`
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/OST/train/img" // HR data root
      , "dataroot_HR_bg": "/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub" // HR images for background, here, SFTGAN uses DIV2K images as the training images for the background category
      , "dataroot_LR": null // if null, generate the LR images on-the-fl
      , "subset_file": null // use a subset of an image folder
      , "use_shuffle": true // shuffle the dataset
      , "n_workers": 8 // number of data load workers
      , "batch_size": 16
      , "HR_size": 96 // cropped HR patch size
      , "use_flip": true // whether use horizontal and vertical flips
      , "use_rot": false // whether use rotations: 90, 190, 270 degrees
    }
    , "val": { // validation dataset configurations
      "name": "val_OST300_part"
      , "mode": "LRHRseg_bg"
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/OST/val/img"
      , "dataroot_LR": null
    }
  }

  , "path": {
    "root": "/home/xtwang/Projects/BasicSR" // root path
    , "pretrain_model_G": "../experiments/pretrained_models/sft_net_ini.pth" // path of the pretrained model
  }

  , "network_G": { // configurations for the network G
    "which_model_G": "sft_arch" // network structures, ref: `https://github.com/xinntao/BasicSR/blob/master/codes/models/networks.py`
  }
  , "network_D": { // configurations for the network D
    "which_model_D": "dis_acd" // network structures, ref: `https://github.com/xinntao/BasicSR/blob/master/codes/models/networks.py`
  }

  , "train": { // training strategies
    "lr_G": 1e-4 // initialized learning rate for G
    , "weight_decay_G": 0
    , "beta1_G": 0.9
    , "lr_D": 1e-4 // initialized learning rate for D
    , "weight_decay_D": 0
    , "beta1_D": 0.9
    , "lr_scheme": "MultiStepLR" // learning rate decay scheme
    , "lr_steps": [50000, 100000, 150000, 200000] // at which steps, decay the learining rate
    , "lr_gamma": 0.5

    , "pixel_criterion": "l1" // "l1" | "l2", pixel criterion
    , "pixel_weight": 0
    , "feature_criterion": "l1" // perceptual criterion (VGG loss)
    , "feature_weight": 1
    , "gan_type": "vanilla" // GAN type
    , "gan_weight": 5e-3

    //for wgan-gp
    // , "D_update_ratio": 1
    // , "D_init_iters": 0
    // , "gp_weigth": 10

    , "manual_seed": 0
    , "niter": 6e5 // total training iteration
    , "val_freq": 2e3 // validation frequency
  }

  , "logger": { // logger configurations
    "print_freq": 200
    , "save_checkpoint_freq": 2e3
  }
}
'''
