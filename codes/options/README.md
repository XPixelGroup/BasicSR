- use **json** file to configure options.
- it will convert json file to python dict.
- support `//` comments and use `null` for None

## configurations
Take `SR.json` for example
```c++
{
  "name": "exp name" // leading 'debug_' will be debug mode
  ,"use_tb_logger": true // use [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
  ,"model":"sr" // model type, ref: 'codes/models/__init__.py'
  ,"scale": 4 // scale factor for SR
  ,"gpu_ids": [0] // useless, please use `export CUDA_VISIBLE_DEVICES=2` in termimal
  
//########################################
//# datasets
//########################################
  ,"datasets": {
    "train": { // "train" | "val" | "test"
      "name": "DIV2K" // dataset name
      ,"mode": "LRHR" // dataset mode, ref: 'codes/data/__init__.py'
      ,"dataroot_HR": "path to HR dataset" // path end with .lmdb will read lmdb file
      ,"dataroot_LR": "path to LR dataset" // if null, generate LR on-the-fly
      ,"subset_file": null // use a subset of an image folder
      ,"use_shuffle": true
      ,"n_workers": 8
      ,"batch_size": 16 
      ,"HR_size": 128 // training patch size
      ,"use_flip": true // horizontal flip
      ,"use_rot": true // rotation
    }
    ,"val": {
      "name": "val_set5"
      ,"mode": "LRHR"
      ,"dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/val_set5/Set5"
      ,"dataroot_LR": "/mnt/SSD/xtwang/BasicSR_datasets/val_set5/Set5_bicLRx4"
    }
  }

  ,"path": {
    "root": "/home/xtwang/Projects/BasicSR" // path of this repo
    ,"pretrain_model_G": null
  }

//########################################
//# define networks
//########################################
  ,"network_G": {
    "which_model_G": "sr_resnet" // model type, ref: 'codes/models/networks.py'
    ,"norm_type": null // "batch" | "instance" | null
    ,"mode": "CNA"
    ,"nf": 64
    ,"nb": 16
    ,"in_nc": 3
    ,"out_nc": 3
    ,"group": 1
  }

//########################################
//# training
//########################################
  ,"train": {
    "manual_seed": 0
    ,"niter": 1e6
    
    // learning rate scheme
    ,"lr_G": 2e-4
    ,"lr_scheme": "MultiStepLR"
    ,"lr_steps": [200000, 400000, 600000, 800000]
    ,"lr_gamma": 0.5
    
    // loss
    ,"pixel_criterion": "l1"
    ,"pixel_weight": 1.0
    ,"val_freq": 5e3
  }

  ,"logger": {
    "print_freq": 200
    ,"save_checkpoint_freq": 5e3
  }
}
```
