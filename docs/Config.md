# Configuration

[English](Config.md) **|** [简体中文](Config_CN.md)

#### Contents

1. [Experiment Name Convention](#Experiment-Name-Convention)
1. [Configuration Explanation](#Configuration-Explanation)
    1. [Training Configuration](#Training-Configuration)
    1. [Testing Configuration](#Testing-Configuration)

## Experiment Name Convention

Taking `001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb` as an example:

- `001`: We usually use index for managing experiments
- `MSRResNet`: Model name, here is Modified SRResNet
- `x4_f64b16`: Import configuration parameters. It means the upsampling ratio is 4; the channel number of middle features is 64; and it uses 16 residual block
- `DIV2K`: Training data is DIV2K
- `1000k`: Total training iteration is 1000k
- `B16G1`: Batch size is 16; one GPU is used for training
- `wandb`: Use wandb logger; the training process has beed uploaded to wandb server

**Note**: If `debug` is in the experiment name, it will enter the debug mode. That is, the program will log and validate more intensively and will not use `tensorboard logger` and `wandb logger`.

## Configuration Explanation

We use yaml files for configuration.

### Training Configuration

Taking [train_MSRResNet_x4.yml](../options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml) as an example:

```yml
####################################
# The following are general settings
####################################
# Experiment name, more details are in [Experiment Name Convention]. If debug in the experiment name, it will enter debug mode
name: 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb
# Model type. Usually the class name defined in the `models` folder
model_type: SRModel
# The scale of the output over the input. In SR, it is the upsampling ratio. If not defined, use 1
scale: 4
# The number of GPUs for training
num_gpu: 1  # set num_gpu: 0 for cpu mode
# Random seed
manual_seed: 0

########################################################
# The following are the dataset and data loader settings
########################################################
datasets:
  # Training dataset settings
  train:
    # Dataset name
    name: DIV2K
    # Dataset type. Usually the class name defined in the `data` folder
    type: PairedImageDataset
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # GT (Ground-Truth) folder path
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    # LQ (Low-Quality) folder path
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
    # template for file name. Usually, LQ files have suffix like `_x4`. It is used for file name mismatching
    filename_tmpl: '{}'
    # IO backend, more details are in [docs/DatasetPreparation.md]
    io_backend:
      # directly read from disk
      type: disk

    # Ground-Truth training patch size
    gt_size: 128
    # Whether to use horizontal flip. Here, flip is for horizontal flip
    use_hflip: true
    # Whether to rotate. Here for rotations with every 90 degree
    use_rot: true

    #### The following are data loader settings
    # Whether to shuffle
    use_shuffle: true
    # Number of workers of reading data for each GPU
    num_worker_per_gpu: 6
    # Total training batch size
    batch_size_per_gpu: 16
    # THe ratio of enlarging dataset. For example, it will repeat 100 times for a dataset with 15 images
    # So that after one epoch, it will read 1500 times. It is used for accelerating data loader
    # since it costs too much time at the start of a new epoch
    dataset_enlarge_ratio: 100

  # validation dataset settings
  val:
    # Dataset name
    name: Set5
    # Dataset type. Usually the class name defined in the `data` folder
    type: PairedImageDataset
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # GT (Ground-Truth) folder path
    dataroot_gt: datasets/Set5/GTmod12
    # LQ (Low-Quality) folder path
    dataroot_lq: datasets/Set5/LRbicx4
    # IO backend, more details are in [docs/DatasetPreparation.md]
    io_backend:
      # directly read from disk
      type: disk

##################################################
# The following are the network structure settings
##################################################
# network g settings
network_g:
  # Architecture type. Usually the class name defined in the `basicsr/archs` folder
  type: MSRResNet
  #### The following arguments are flexible and can be obtained in the corresponding doc
  # Channel number of inputs
  num_in_ch: 3
  # Channel number of outputs
  num_out_ch: 3
  # Channel number of middle features
  num_feat: 64
  # block number
  num_block: 16
  # upsampling ratio
  upscale: 4

#########################################################
# The following are path, pretraining and resume settings
#########################################################
path:
  # Path for pretrained models, usually end with pth
  pretrain_network_g: ~
  # Whether to load pretrained models strictly, that is the corresponding parameter names should be the same
  strict_load_g: true
  # Path for resume state. Usually in the `experiments/exp_name/training_states` folder
  # This argument will over-write the `pretrain_network_g`
  resume_state: ~


#####################################
# The following are training settings
#####################################
train:
  # Optimizer settings
  optim_g:
    # Optimizer type
    type: Adam
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # Learning rate
    lr: !!float 2e-4
    weight_decay: 0
    # beta1 and beta2 for the Adam
    betas: [0.9, 0.99]

  # Learning rate scheduler settings
  scheduler:
    # Scheduler type
    type: CosineAnnealingRestartLR
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # Cosine Annealing periods
    periods: [250000, 250000, 250000, 250000]
    # Cosine Annealing restart weights
    restart_weights: [1, 1, 1, 1]
    # Cosine Annealing minimum learning rate
    eta_min: !!float 1e-7

  # Total iterations for training
  total_iter: 1000000
  # Warm up iterations. -1 indicates no warm up
  warmup_iter: -1

  #### The following are loss settings
  # Pixel-wise loss options
  pixel_opt:
    # Loss type. Usually the class name defined in the `basicsr/models/losses` folder
    type: L1Loss
    # Loss weight
    loss_weight: 1.0
    # Loss reduction mode
    reduction: mean


#######################################
# The following are validation settings
#######################################
val:
  # validation frequency. Validate every 5000 iterations
  val_freq: !!float 5e3
  # Whether to save images during validation
  save_img: false

  # Metrics in validation
  metrics:
    # Metric name. It can be arbitrary
    psnr:
      # Metric type. Usually the function name defined in the`basicsr/metrics` folder
      type: calculate_psnr
      #### The following arguments are flexible and can be obtained in the corresponding doc
      # Whether to crop border during validation
      crop_border: 4
      # Whether to convert to Y(CbCr) for validation
      test_y_channel: false

########################################
# The following are the logging settings
########################################
logger:
  # Logger frequency
  print_freq: 100
  # The frequency for saving checkpoints
  save_checkpoint_freq: !!float 5e3
  # Whether to tensorboard logger
  use_tb_logger: true
  # Whether to use wandb logger. Currently, wandb only sync the tensorboard log. So we should also turn on tensorboard when using wandb
  wandb:
    # wandb project name. Default is None, that is not using wandb.
    # Here, we use the basicsr wandb project: https://app.wandb.ai/xintao/basicsr
    project: basicsr
    # If resuming, wandb id could automatically link previous logs
    resume_id: ~

################################################
# The following are distributed training setting
# Only require for slurm training
################################################
dist_params:
  backend: nccl
  port: 29500
```

### Testing Configuration

Taking [test_MSRResNet_x4.yml](../options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml) as an example:

```yml
# Experiment name
name: 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb
# Model type. Usually the class name defined in the `models` folder
model_type: SRModel
# The scale of the output over the input. In SR, it is the upsampling ratio. If not defined, use 1
scale: 4
# The number of GPUs for testing
num_gpu: 1  # set num_gpu: 0 for cpu mode

########################################################
# The following are the dataset and data loader settings
########################################################
datasets:
  # Testing dataset settings. The first testing dataset
  test_1:
    # Dataset name
    name: Set5
    # Dataset type. Usually the class name defined in the `data` folder
    type: PairedImageDataset
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # GT (Ground-Truth) folder path
    dataroot_gt: datasets/Set5/GTmod12
    # LQ (Low-Quality) folder path
    dataroot_lq: datasets/Set5/LRbicx4
    # IO backend, more details are in [docs/DatasetPreparation.md]
    io_backend:
      # directly read from disk
      type: disk
  # Testing dataset settings. The second testing dataset
  test_2:
    name: Set14
    type: PairedImageDataset
    dataroot_gt: datasets/Set14/GTmod12
    dataroot_lq: datasets/Set14/LRbicx4
    io_backend:
      type: disk
  # Testing dataset settings. The third testing dataset
  test_3:
    name: DIV2K100
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_valid_HR
    dataroot_lq: datasets/DIV2K/DIV2K_valid_LR_bicubic/X4
    filename_tmpl: '{}x4'
    io_backend:
      type: disk

##################################################
# The following are the network structure settings
##################################################
# network g settings
network_g:
  # Architecture type. Usually the class name defined in the `basicsr/archs` folder
  type: MSRResNet
  #### The following arguments are flexible and can be obtained in the corresponding doc
  # Channel number of inputs
  num_in_ch: 3
  # Channel number of outputs
  num_out_ch: 3
  # Channel number of middle features
  num_feat: 64
  # block number
  num_block: 16
  # upsampling ratio
  upscale: 4
  upscale: 4

#################################################
# The following are path and pretraining settings
#################################################
path:
  ## Path for pretrained models, usually end with pth
  pretrain_network_g: experiments/001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb/models/net_g_1000000.pth
  # Whether to load pretrained models strictly, that is the corresponding parameter names should be the same
  strict_load_g: true

##########################################################
# The following are validation settings (Also for testing)
##########################################################
val:
  # Whether to save images during validation
  save_img: true
  # Suffix for saved images. If None, use exp name
  suffix: ~

  # Metrics in validation
  metrics:
    # Metric name. It can be arbitrary
    psnr:
      # Metric type. Usually the function name defined in the`basicsr/metrics` folder
      type: calculate_psnr
      #### The following arguments are flexible and can be obtained in the corresponding doc
      # Whether to crop border during validation
      crop_border: 4
      # Whether to convert to Y(CbCr) for validation
      test_y_channel: false
    # Another metric
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: false
```
