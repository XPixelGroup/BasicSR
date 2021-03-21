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
export CUDA_VISIBLE_DEVICES=0
python basicsr/test.py -opt options/test/ESRGAN/test_additional_ESRGAN_gray_x4_02.yml 
