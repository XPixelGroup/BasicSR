#!/bin/bash
#PBS -q h-small
#PBS -l select=1:mpiprocs=1:ompthreads=1
#PBS -W group_list=gp14
#PBS -l walltime=48:00:00
#PBS -j oe
cd $PBS_O_WORKDIR
module load anaconda3/2020.07 cuda10/10.2.89 gnu/gcc_7.3.0
export PYTHONUSERBASE="/lustre/gp14/p14008/local"
cd ../../BasicSR
export PYTHONPATH="./:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/ESRGAN/train_ESRGAN_gray_x4_02_2.yml --launcher pytorch
