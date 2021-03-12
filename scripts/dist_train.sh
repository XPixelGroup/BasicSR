#!/usr/bin/env bash

# useage

GPUS=$1
CONFIG=$2
PORT=${PORT:-4321}

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    basicsr/train.py -opt $CONFIG --launcher pytorch ${@:3}