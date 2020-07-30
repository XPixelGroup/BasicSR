# 训练和测试
[English](TrainTest.md) | [简体中文](TrainTest_CN.md)

所有的命令都在 `BasicSR` 的根目录下运行. <br>
一般来说, 训练和测试都有以下的步骤:
1. 准备数据. 参见 [DatasetPreparation_CN.md](DatasetPreparation_CN.md)
1. 修改Config文件. Config文件在 `options` 目录下面. 具体的Config配置含义, 可参考 [Config说明](Config_CN.md)
1. [Optional] 如果是测试或需要预训练, 则需下载预训练模型, 参见 [模型库](ModelZoo_CN.md)
1. 运行命令. 根据需要，使用 [训练命令](#训练命令) 或 [测试命令](#测试命令)

#### 目录
1. [](#数据存储形式)
    1. [如何使用](#如何使用)
    1. [如何实现](#如何实现)
    1. [LMDB具体说明](#LMDB具体说明)
1. [图像数据](#图像数据)
    1. [DIV2K](#DIV2K)
    1. [其他常见图像超分数据集](#其他常见图像超分数据集)
1. [视频帧数据](#视频帧数据)
    1. [REDS](#REDS)
    1. [Vimeo90K](#Vimeo90K)

## 训练命令
```bash
############################
# Single GPU training
############################
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml

############################
# Single GPU testing
############################
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml


############################
# Distributed training
############################
# 1 GPU
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNetx4.yml --launcher pytorch

# 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher pytorch

# 8 GPUs
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/train_EDVR_M.yml --launcher pytorch

############################
# Distributed testing
############################
# 1 GPU
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/test_TOF_official.yml --launcher pytorch

# 4 GPUs

# 8 GPUs

```

## 测试命令
```
############################
# Slurm training
############################
# 1 GPU
GLOG_vmodule=MemcachedClient=-1 srun -p partition --mpi=pmi2 --job-name=999 --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 python -u basicsr/train.py -opt options/train/train_SRResNet.yml --launcher="slurm"

# 4 GPUs
GLOG_vmodule=MemcachedClient=-1 srun -p partition --mpi=pmi2 --job-name=EDVRwoTSA --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --kill-on-bad-exit=1 python -u basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher="slurm"

# 8 GPUs
GLOG_vmodule=MemcachedClient=-1 srun -p partition --mpi=pmi2 --job-name=EDVRwoTSA --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 python -u basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml  --launcher="slurm"

############################
# Slurm testing
############################
# 1 GPU
GLOG_vmodule=MemcachedClient=-1 srun -p partition --mpi=pmi2 --job-name=test --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 python -u basicsr/test.py -opt options/test/test_video_recurrent.yml --launcher="slurm"

# 4 GPUs

# 8 GPUs

```


## Tensorboard

```sh
tensorboard --logdir tb_logger --port 5500 --bind_all
```


#### Train ESRGAN (SRGAN) models
We use a PSNR-oriented pre-trained SR model to initialize the parameters for better quality and faster convergence.

1. Prepare datasets, usually the DIV2K dataset. More details are in [DATASETS.md](docs/DATASETS.md).
1. Prepare the PSNR-oriented pre-trained model. You can use the `RRDB_PSNR_x4.pth` as the pretrained model.
1. Modify the configuration file  `options/train/train_ESRGAN.yml`
1. Run the command: `python train.py -opt options/train/train_ESRGAN.yml`
# Video SR
## Testing
1. Download the pre-trained model from [Model Zoo](docs/ModelZoo.md).
2. Download the [testing datasts](https://drive.google.com/open?id=1EwEXDYImflknnZS0rJy8gD1zOi8ZgEXa).
3. Run `test_Vid4_REDS4_with_GT.py`

You can also test the DUF and TOFlow models.

## Training
We use distributed training with eight GPUs.

`python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt options/train/train_EDVR_woTSA_M.yml --launcher pytorch`

We provide the training configurations for a moderate model with channel size = 64 and back residual block number = 10.

1. Train with the config `train_EDVR_woTSA_M.yml`. [[ Example of training log](https://drive.google.com/open?id=1_qRwexMyKQbLSfxA8-TKroA0LEzn3s4Y) ], [ [Pre-trained model](https://drive.google.com/open?id=1ProhBk4FtSb5pWT2g5PhVH9GecZB_T_6) ]
2. Train with the config `train_EDVR_M.yml`, whose initialization is from the model of Step 1. [ [Example of training log](https://drive.google.com/open?id=1JmlbKxF8Xz9hb0hR5sCKXYMjvu15x04J) ], [ [Pre-trained model](https://drive.google.com/open?id=1PVlzQ2UiGBzCvZxIFQo6EA5uf2LDVJwy) ]
