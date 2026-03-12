#!/bin/bash

GEN_DIR=${1:-work_dirs/test_video_submit}
OUT_DIR=${2:-work_dirs/test_video_submit_fvd}

echo "============================================="
echo " Step 1: Generate features for original data "
echo "============================================="
python tools/fvd_score/get_fvd_features_nusc.py \
    --data_info data/nuscenes/workshop/nuscenes_interp_12Hz_infos_track2_eval.pkl \
    --out_dir "$OUT_DIR"

echo "============================================="
echo " Step 2: Generate features for generated data"
echo "============================================="
python tools/fvd_score/get_fvd_features_gen.py \
    --gen_dir "$GEN_DIR" \
    --out_dir "$OUT_DIR"

echo "============================================="
echo " Step 3: FVD calculation "
echo "============================================="
python tools/fvd_score/fvd_from_npy.py \
    "$OUT_DIR/fvd_feats_ori.npy" \
    "$OUT_DIR/fvd_feats_gen.npy"
