# 训练和测试

[English](TrainTest.md) **|** [简体中文](TrainTest_CN.md)

所有的命令都在 `BasicSR` 的根目录下运行. <br>
一般来说, 训练和测试都有以下的步骤:

1. 准备数据. 参见 [DatasetPreparation_CN.md](DatasetPreparation_CN.md)
1. 修改Config文件. Config文件在 `options` 目录下面. 具体的Config配置含义, 可参考 [Config说明](Config_CN.md)
1. [Optional] 如果是测试或需要预训练, 则需下载预训练模型, 参见 [模型库](ModelZoo_CN.md)
1. 运行命令. 根据需要，使用 [训练命令](#训练命令) 或 [测试命令](#测试命令)

#### 目录

1. [训练命令](#训练命令)
    1. [单GPU训练](#单GPU训练)
    1. [分布式(多卡)训练](#分布式训练)
    1. [Slurm训练](#Slurm训练)
1. [测试命令](#测试命令)
    1. [单GPU测试](#单GPU测试)
    1. [分布式(多卡)测试](#分布式测试)
    1. [Slurm测试](#Slurm测试)

## 训练命令

### 单GPU训练

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml

### 分布式训练

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher pytorch

或者

> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> ./scripts/dist_train.sh 8 options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher pytorch

或者

> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> ./scripts/dist_train.sh 4 options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml

### Slurm训练

[Slurm介绍](https://slurm.schedmd.com/quickstart.html)

**1 GPU**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=MSRResNetx4 --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml --launcher="slurm"

**4 GPUs**


> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=EDVRMwoTSA --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --kill-on-bad-exit=1 \\\
> python -u basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher="slurm"

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=EDVRMwoTSA --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher="slurm"

## 测试命令

### 单GPU测试

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml

### 分布式测试

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher pytorch

或者

> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> ./scripts/dist_test.sh 8 options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml  --launcher pytorch

或者

> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> ./scripts/dist_test.sh 4 options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml

### Slurm测试

[Slurm介绍](https://slurm.schedmd.com/quickstart.html)

**1 GPU**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=test --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml --launcher="slurm"

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=test --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --kill-on-bad-exit=1 \\\
> python -u basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher="slurm"

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=test --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher="slurm"
