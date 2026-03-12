NUM_PROCESSES=$1
MODE=$2
IMG_MODEL=${3:-pretrained/x-scene-img_224x400}
VIDEO_MODEL=${4:-pretrained/x-scene-video_224x400}
OUT_DIR=${5:-work_dirs/test_video_submit}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-28600}
LAUNCH_PARAM=${LAUNCH_PARAM:-"--num_processes ${NUM_PROCESSES} --main_process_port ${MAIN_PROCESS_PORT} --num_machines 1"}

if [ "$MODE" = "gen" ]; then
    accelerate launch --gpu_ids all ${LAUNCH_PARAM} \
        tools/test_video.py \
        --model_img ${IMG_MODEL} \
        --model_video ${VIDEO_MODEL} \
        --output ${OUT_DIR} \
        --overlap_condition

elif [ "$MODE" = "submit" ]; then
    accelerate launch --gpu_ids all ${LAUNCH_PARAM} \
        tools/test_video_submit.py \
        --model_img ${IMG_MODEL} \
        --model_video ${VIDEO_MODEL} \
        --output ${OUT_DIR}
else
    echo "Error: Invalid mode '$MODE'. Please use 'gen' or 'submit'."
fi