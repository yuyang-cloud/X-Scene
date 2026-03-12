# Prepare Dataset
We prepare the nuScenes dataset similar to [MagicDrive's instructions](https://github.com/cure-lab/MagicDrive?tab=readme-ov-file#prepare-data). Specifically,


**1. Download the nuScenes dataset from the [website](https://www.nuscenes.org/nuscenes) and put them in `data/`. You should have these files:**

```bash
data/nuscenes
├── maps
├── mini
├── samples
├── sweeps
├── v1.0-mini
└── v1.0-trainval
```

**2. Generate `ann_file` for keyframes (samples) by:**
```shell
python -m tools.create_data nuscenes \
--root-path data/nuscenes --out-dir data/nuscenes/nuscenes_mmdet3d-keyframes \
--extra-tag nuscenes --only_info
```

You will get these files:
```bash
data/nuscenes/nuscenes_mmdet3d-keyframes
├── nuscenes_infos_train.pkl
└── nuscenes_infos_val.pkl
```

**3. Generate `ann_file` for 12Hz (sweeps) by:**
- Firstly, follow [ASAP](https://github.com/JeffWang987/ASAP/blob/main/docs/prepare_data.md) to generate interp annotations for nuScenes. 

    **Note**: The following codes in ASAP need to be modified:
    
    - In `sAP3D/nusc_annotation_generator.py`, please comment [line357](https://github.com/JeffWang987/ASAP/blob/52316629f2a87ef2ef5bbc634d33e9544b5e39a7/sAP3D/nusc_annotation_generator.py#L357), and modify [line101](https://github.com/JeffWang987/ASAP/blob/52316629f2a87ef2ef5bbc634d33e9544b5e39a7/sAP3D/nusc_annotation_generator.py#L101) to `val_scene_ids = splits['val'] + splits['train']`.
    
    - Modify the dataset path in `scripts/ann_generator.sh` to your custom dataset path.

    Then, you can run the following command in ASAP root:
    ```
    bash scripts/ann_generator.sh 12 --ann_strategy 'interp' 
    ```
    (Optional) Generate advanced annotations for sweeps. (We do not observe major difference between interp and advanced. You can refer to the implementation of [ASAP](https://github.com/JeffWang987/ASAP/blob/main/docs/prepare_data.md). This step can be skipped.)

    Rename the generated folder to `interp_12Hz_trainval` and move it into your nuScenes dataset root `data/nuscenes`.
    
- Use the following command to generate ann_file with 12hz.
    ```
    python tools/create_data.py nuscenes --root-path data/nuscenes --out-dir data/nuscenes/nuscenes_mmdet3d-12Hz \
    --extra-tag nuscenes_interp_12Hz --max-sweeps -1 --version interp_12Hz_trainval
    ```

You will get these files:
```bash
data/nuscenes/nuscenes_mmdet3d-12Hz
├── nuscenes_interp_12Hz_infos_train.pkl
└── nuscenes_interp_12Hz_infos_val.pkl
```

**4. (Optional but recommended) We recommend generating cache files in `.h5` format of the BEV map to speed up the data loading process.**
```
# generate map cache for val
python tools/prepare_map_aux.py +process=val +subfix=12Hz_interp

# generate map cache for train
python tools/prepare_map_aux.py +process=train +subfix=12Hz_interp
```

You will get these files:
```bash
data/nuscenes/nuscenes_map_aux_12Hz_interp
├── train_200x200_12Hz_interp.h5 (264G)
└── val_200x200_12Hz_interp.h5 (53G)
```

**5. Download our pre-generated scene descriptions for each keyframe from [here](https://huggingface.co/yuyang-cloud/X-Scene/blob/main/description.zip). Unzip the file and place the contents into the `data/nuscenes/` directory.**

Your directory structure should look like this:
```bash
data/nuscenes/description
├── scene-0001              # Scene Token
│   ├── 1e19d0a5189b46f4b62aa47508f2983e # Frame Token
│   │   └── description.json
│   ├── 2acfc4b99f084778bc484fa1eef1b79f # Frame Token
│   │   └── description.json
│   └── ...
├── scene-0002
└── ...
```

**6. Download the `nuscenes_interp_12Hz_infos_track2_eval.pkl` and `nuscenes_interp_12Hz_infos_track2_eval_long.pkl` from [w-coda2024/track2](https://coda-dataset.github.io/w-coda2024/track2/) for 16-frame videos FVD benchmark evaluation.**

You will get these files:
```bash
data/nuscenes/workshop
├── nuscenes_interp_12Hz_infos_track2_eval.pkl
└── nuscenes_interp_12Hz_infos_track2_eval_long.pkl
```

**7. Prepare Occupancy annotations as defined in [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D):**

<div align="center">
  
| Type |  Info |
| :----: | :----: |
| train           | 600 |
| val             | 150 |
| test            | 250 |
| cameras         | 6 |
| voxel size      | [0.4m, 0.4m, 0.4m] |
| range           | [-40m, -40m, -1m, 40m, 40m, 5.4m] |
| volume size     | [200, 200, 16]|
| classes        | 0 - 17 |
</div>

1) Download the occupancy annotations `gts.tar.gz` from [here](https://drive.google.com/file/d/17HubGsfioQr1d_39VwVPXelobAFo4Xqh/view?usp=drive_link), then unzip it to the `data/nuscenes` directory.

2) Download the train/val pickle files and put them in `data/nuscenes/`

    [nuscenes_infos_train_temporal_v3_scene.pkl](https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/)

    [nuscenes_infos_val_temporal_v3_scene.pkl](https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/)

3) You will get the folder structure:
    ```
    data/nuscenes/
    ├── gts/
    |   |── scenes-0001/
    |   | ...
    │
    │──nuscenes_infos_train_temporal_v3_scene.pkl
    │──nuscenes_infos_val_temporal_v3_scene.pkl
    ```
