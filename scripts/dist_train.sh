NUM_PROCESSES=$1
CONFIG_NAME=$2
PY_ARGS=${@:3}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29600}
LAUNCH_PARAM=${LAUNCH_PARAM:-"--num_processes ${NUM_PROCESSES} --main_process_port ${MAIN_PROCESS_PORT} --num_machines 1"}

accelerate launch --mixed_precision fp16 --gpu_ids all \
    ${LAUNCH_PARAM} tools/train.py --config-name ${CONFIG_NAME} ${PY_ARGS}