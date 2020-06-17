# single GPU training
python basicsr/train.py -opt options/train/train_SRResNet.yml

# single GPU testing
python basicsr/test.py -opt options/test/test_SRResNet.yml


# Distributed training, pytorch
# distributed training, 1 GPUs
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/train_SRResNet.yml --launcher pytorch

# distributed training, 8 GPUs
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/train_EDVR_M.yml --launcher pytorch

# distributed testing. 1GPUs
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/test_TOF_official.yml --launcher pytorch

# distributed testing. 4GPUs
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/test_EDVR.yml --launcher pytorch

# Distributed training, slurm
# slurm, 1 GPU
GLOG_vmodule=MemcachedClient=-1 srun -p mediaf --mpi=pmi2 --job-name=999 --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 python -u basicsr/train.py -opt options/train/train_SRResNet.yml --launcher="slurm"

# slurm, 8 GPUs
GLOG_vmodule=MemcachedClient=-1 srun -p mediaf --mpi=pmi2 --job-name=000 --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 python -u basicsr/train.py -opt options/train/train_EDVRM_woTSA.yml --launcher="slurm"

# slurm, 4 GPUs
GLOG_vmodule=MemcachedClient=-1 srun -p mediaf --mpi=pmi2 --job-name=999 --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --kill-on-bad-exit=1 python -u basicsr/train.py -opt options/train/train_EDVRM_woTSA.yml --launcher="slurm"

# test, slurm, 1 GPU
GLOG_vmodule=MemcachedClient=-1 srun -p mediaf --mpi=pmi2 --job-name=test --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 python -u basicsr/test.py -opt options/test/test_video_recurrent.yml --launcher="slurm"
