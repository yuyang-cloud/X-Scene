CONFIG_NAME=$1
PY_ARGS=${@:2}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-26500}
LAUNCH_PARAM=${LAUNCH_PARAM:-"--num_processes 1 --main_process_port ${MAIN_PROCESS_PORT} --num_machines 1"}

accelerate launch --mixed_precision fp16 --gpu_ids all \
    ${LAUNCH_PARAM} tools/train.py --config-name ${CONFIG_NAME} ${PY_ARGS}

# bash scripts/debug.sh config_occ +exp=occ runner=debug runner.validation_before_run=true