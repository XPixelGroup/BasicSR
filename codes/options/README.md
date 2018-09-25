- Use **json** file to configure options.
- Convert json file to python dict.
- Support `//` comments and use `null` for `None`

# Configurations
1. [train_sr.json](#train_sr_json)
1. [train_esrgan.json](#train_esrgan_json)
1. [train_sftgan.json](#train_sftgan_json)

## train_sr_json
```c++
{
  "name": "debug_001_RRDB_PSNR_x4_DIV2K" //  please remove "debug_" during training
  , "use_tb_logger": true
  , "model":"sr"
  , "scale": 4
  , "gpu_ids": [0]

  , "datasets": {
    "train": {
      "name": "DIV2K"
      , "mode": "LRHR"
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb"
      , "dataroot_LR": "/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub_bicLRx4.lmdb"
      , "subset_file": null
      , "use_shuffle": true
      , "n_workers": 8
      , "batch_size": 16
      , "HR_size": 128 // 128 | 192
      , "use_flip": true
      , "use_rot": true
    }
    , "val": {
      "name": "val_set5"
      , "mode": "LRHR"
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/val_set5/Set5"
      , "dataroot_LR": "/mnt/SSD/xtwang/BasicSR_datasets/val_set5/Set5_bicLRx4"
    }
  }

  , "path": {
    "root": "/home/xtwang/Projects/BasicSR"
    , "pretrain_model_G": null
  }

  , "network_G": {
    "which_model_G": "RRDB_net" // RRDB_net | sr_resnet
    , "norm_type": null
    , "mode": "CNA"
    , "nf": 64
    , "nb": 23
    , "in_nc": 3
    , "out_nc": 3
    , "gc": 32
    , "group": 1
  }

  , "train": {
    "lr_G": 2e-4
    , "lr_scheme": "MultiStepLR"
    , "lr_steps": [200000, 400000, 600000, 800000]
    , "lr_gamma": 0.5

    , "pixel_criterion": "l1"
    , "pixel_weight": 1.0
    , "val_freq": 5e3

    , "manual_seed": 0
    , "niter": 1e6
  }

  , "logger": {
    "print_freq": 200
    , "save_checkpoint_freq": 5e3
  }
}
```
## train_esrgan_json
```c++
{
  "name": "debug_002_RRDB_ESRGAN_x4_DIV2K" //  please remove "debug_" during training
  , "use_tb_logger": true
  , "model":"srragan"
  , "scale": 4
  , "gpu_ids": [0]

  , "datasets": {
    "train": {
      "name": "DIV2K"
      , "mode": "LRHR"
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb"
      , "dataroot_LR": "/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub_bicLRx4.lmdb"
      , "subset_file": null
      , "use_shuffle": true
      , "n_workers": 8
      , "batch_size": 16
      , "HR_size": 128
      , "use_flip": true
      , "use_rot": true
    }
    , "val": {
      "name": "val_set14_part"
      , "mode": "LRHR"
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/val_set14_part/Set14"
      , "dataroot_LR": "/mnt/SSD/xtwang/BasicSR_datasets/val_set14_part/Set14_bicLRx4"
    }
  }

  , "path": {
    "root": "/home/xtwang/Projects/BasicSR"
    , "pretrain_model_G": "../experiments/pretrained_models/RRDB_PSNR_x4.pth"
  }

  , "network_G": {
    "which_model_G": "RRDB_net" // RRDB_net | sr_resnet
    , "norm_type": null
    , "mode": "CNA"
    , "nf": 64
    , "nb": 23
    , "in_nc": 3
    , "out_nc": 3
    , "gc": 32
    , "group": 1
  }
  , "network_D": {
    "which_model_D": "discriminator_vgg_128"
    , "norm_type": "batch"
    , "act_type": "leakyrelu"
    , "mode": "CNA"
    , "nf": 64
    , "in_nc": 3
  }

  , "train": {
    "lr_G": 1e-4
    , "weight_decay_G": 0
    , "beta1_G": 0.9
    , "lr_D": 1e-4
    , "weight_decay_D": 0
    , "beta1_D": 0.9
    , "lr_scheme": "MultiStepLR"
    , "lr_steps": [50000, 100000, 200000, 300000]
    , "lr_gamma": 0.5

    , "pixel_criterion": "l1"
    , "pixel_weight": 1e-2
    , "feature_criterion": "l1"
    , "feature_weight": 1
    , "gan_type": "vanilla"
    , "gan_weight": 5e-3

    //for wgan-gp
    // , "D_update_ratio": 1
    // , "D_init_iters": 0
    // , "gp_weigth": 10

    , "manual_seed": 0
    , "niter": 5e5
    , "val_freq": 5e3
  }

  , "logger": {
    "print_freq": 200
    , "save_checkpoint_freq": 5e3
  }
}
```
## train_sftgan_json

```c++
{
  "name": "debug_003_SFTGANx4_OST" //  please remove "debug_" during training
  , "use_tb_logger": false
  , "model": "sftgan"
  , "scale": 4
  , "gpu_ids": [0]

  , "datasets": {
    "train": {
      "name": "OST"
      , "mode": "LRHRseg_bg"
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/OST/train/img"
      , "dataroot_HR_bg": "/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub"
      , "dataroot_LR": null
      , "subset_file": null
      , "use_shuffle": true
      , "n_workers": 8
      , "batch_size": 16
      , "HR_size": 96
      , "use_flip": true
      , "use_rot": false
    }
    , "val": {
      "name": "val_OST300_part"
      , "mode": "LRHRseg_bg"
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/OST/val/img"
      , "dataroot_LR": null
    }
  }

  , "path": {
    "root": "/home/xtwang/Projects/BasicSR"
    , "pretrain_model_G": "../experiments/pretrained_models/sft_net_ini.pth"
  }

  , "network_G": {
    "which_model_G": "sft_arch"
  }
  , "network_D": {
    "which_model_D": "dis_acd"
  }

  , "train": {
    "lr_G": 1e-4
    , "weight_decay_G": 0
    , "beta1_G": 0.9
    , "lr_D": 1e-4
    , "weight_decay_D": 0
    , "beta1_D": 0.9
    , "lr_scheme": "MultiStepLR"
    , "lr_steps": [50000, 100000, 150000, 200000]
    , "lr_gamma": 0.5

    , "pixel_criterion": "l1"
    , "pixel_weight": 0
    , "feature_criterion": "l1"
    , "feature_weight": 1
    , "gan_type": "vanilla"
    , "gan_weight": 5e-3

    //for wgan-gp
    // , "D_update_ratio": 1
    // , "D_init_iters": 0
    // , "gp_weigth": 10

    , "manual_seed": 0
    , "niter": 6e5
    , "val_freq": 2e3
  }

  , "logger": {
    "print_freq": 200
    , "save_checkpoint_freq": 2e3
  }
}
'''
