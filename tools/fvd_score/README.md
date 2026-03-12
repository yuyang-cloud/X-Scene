### Get Started

Setup python environment and pre-trained weights.

- Download [i3d_pretrained_400.pt](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI) and put it in `pretrained/fvd/videogpt`

Ensure that the directory `work_dirs/test_video_submit` contains **900 subfolders**, with each subfolder holding **six generated video files** in the format *_CAM_*.mp4.

### Test FVD

Command:

```bash
# step 1: generate features for original data
python get_fvd_features_nusc.py \
	--data_info data/nuscenes/workshop/nuscenes_interp_12Hz_infos_track2_eval.pkl \
	--out_dir work_dirs/test_video_submit_fvd

# step 2: generate features for generated data
python get_fvd_features_gen.py \
	--gen_dir work_dirs/test_video_submit \
    --out_dir work_dirs/test_video_submit_fvd

# step 3: fvd calculation
python fvd_from_npy.py \
	work_dirs/test_video_submit_fvd/fvd_feats_ori.npy \
	work_dirs/test_video_submit_fvd/fvd_feats_gen.npy 
```
