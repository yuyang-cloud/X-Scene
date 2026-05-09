#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-16842}

torchrun --nproc_per_node=$GPUS --master_port=$PORT train_f3d.py
