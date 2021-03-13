#!/usr/bin/env bash

# usage

GPUS=$1
CONFIG=$2
PORT=${PORT:-4321}

if [ $# -le 3 ] ;then
    echo "usage:"
    echo "./scripts/dist_train.sh [number of gpu] [path to option file]"
fi

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    basicsr/train.py -opt $CONFIG --launcher pytorch ${@:3}