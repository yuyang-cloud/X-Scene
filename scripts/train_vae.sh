#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-10086}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_vae_main.py "${@:2}"