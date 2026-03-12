NUM_PROCESSES=$1
MODE=$2
CONFIG_NAME=$3
RESUME=$4
TASK_ID=${5:-occ}   # default: occ
PY_ARGS=${@:6}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-28600}
LAUNCH_PARAM=${LAUNCH_PARAM:-"--num_processes ${NUM_PROCESSES} --main_process_port ${MAIN_PROCESS_PORT} --num_machines 1"}

if [ "$MODE" = "debug" ]; then
    echo "Running in DEBUG mode (tools/test.py for visualization/small validation)"
    if [ -z "$PY_ARGS" ]; then
        PY_ARGS="+exp=occ runner=8gpus_occ"
    fi
    accelerate launch --mixed_precision fp16 --gpu_ids all \
        ${LAUNCH_PARAM} tools/test.py --config-name ${CONFIG_NAME} resume_from_checkpoint=${RESUME} \
        task_id=${TASK_ID} ${PY_ARGS}

elif [ "$MODE" = "batch" ]; then
    echo "Running in BATCH mode (perception/data_prepare/val_set_gen.py for full validation set generation)"
    if [ -z "$PY_ARGS" ]; then
        PY_ARGS="+exp=occ +fid=data_gen_occ"
    fi
    accelerate launch --mixed_precision fp16 --gpu_ids all \
        ${LAUNCH_PARAM} perception/data_prepare/val_set_gen.py --config-name ${CONFIG_NAME} resume_from_checkpoint=${RESUME} \
        task_id=${TASK_ID} ${PY_ARGS}

else
    echo "Error: Invalid mode '$MODE'. Please use 'debug' or 'batch'."
    exit 1
fi