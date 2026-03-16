## Pretrained Weights
We use the pre-trained weights of 
[stable-diffusion-v2-1](https://huggingface.co/Manojb/stable-diffusion-2-1-base) to initialize the diffusion model, and [T5-Large](https://huggingface.co/google-t5/t5-large) as the text encoder.

We assume you put them at `${ROOT}/pretrained/` as follows:

```
${ROOT}/pretrained/
        └── stable-diffusion-v2-1/
        └── t5-large/
```

## OccGen Train & Test
### Train 
**Stage-1: Train the Occupancy VAE:**
```bash
bash scripts/train_vae.sh 8 --dataset Occ3D-nuScenes --save_path work_dirs/vae
```

After completing the training, rename the checkpoint directory with the highest mIoU to `work_dirs/vae/best_model`. This will serve as the frozen weights for the VAE in the subsequent training process.
```bash
mv work_dirs/vae/${epoch}_miou=${miou}_model work_dirs/vae/best_model
```

**Stage-2: Save Triplane:**
```bash
python xscene/occ_vae/save_triplane.py
```

You will get these files:
```bash
data/nuscenes/nuscenes_triplane
└── <scene_token>/
    └── triplane/
        ├── <frame_token_1>.npy
        ├── <frame_token_2>.npy
        └── ...
```

**Stage-3: Train the Occupancy Diffusion Model:**
```bash
bash scripts/dist_train.sh 8 config_occ +exp=occ runner=8gpus_occ
```

Besides, we provide debug config to test your environment and data loading process:
```bash
bash scripts/debug.sh config_occ +exp=occ runner=debug runner.validation_before_run=true
```

### Test
We support two modes: `debug` (validation on a small subset) and `batch` (generation on the full validation set).

**Debug Mode (Subset Validation): Runs [tools/test.py](../tools/test.py)**
```bash
# bash scripts/dist_test_occ.sh <NUM_GPUS> debug <CONFIG_NAME> <CKPT_PATH> [TASK_ID] [PY_ARGS]
bash scripts/dist_test_occ.sh 1 debug test_config_occ ${RESUME_CKPT} occ +exp=occ runner=8gpus_occ
```

**Batch Generation Mode (Full Validation Set): Runs [perception/data_prepare/val_set_gen.py](../perception/data_prepare/val_set_gen.py)**
```bash
# bash scripts/dist_test_occ.sh <NUM_GPUS> batch <CONFIG_NAME> <CKPT_PATH> [TASK_ID] [PY_ARGS]
bash scripts/dist_test_occ.sh 8 batch test_config_occ ${RESUME_CKPT} occ +exp=occ +fid=data_gen_occ
```

## ImgGen Train & Test
### Train 
**Stage-1: Prepare Occ Rendered Map (Optional):**
We provide two ways to prepare the occupancy rendered maps (semantic and depth maps).

- **Option 1: Generate from scratch**
    ```bash
    python tools/gs_render/render_train_condition.py
    ```

- **Option 2: Download pre-processed data:**
    Download our pre-processed [`OccRenderMap`](https://huggingface.co/datasets/yuyang-cloud/X-Scene/tree/main/occ_render_map). Merge the split files, extract the archive, and place the contents under `data/nuscenes/`.

You will get these files:
```bash
data/nuscenes/occ_render_map
└── <scene_name>/
    └── <frame_token>/
        ├── semantic.npz
        └── depth_data.npz
```

**Stage-2: Train the Image Diffusion Model:**
We recommend a progressive training strategy, starting from low resolution and moving to higher resolutions.

* **Step 1: Low-Resolution Training ($224 \times 400$):**
```bash
bash scripts/dist_train.sh 8 config_img +exp=224x400 runner=8gpus_img
```

* **Step 2: Mid-Resolution Fine-tuning ($336 \times 600$):**
Update `config_img.yaml`: `resume_from_checkpoint` = path to 224x400 checkpoint, `resume_only_model=true`
```bash
bash scripts/dist_train.sh 8 config_img +exp=336x600 runner=8gpus_img
```

* **Step 3: High-Resolution Fine-tuning ($448 \times 800$):**
Update `config_img.yaml`: `resume_from_checkpoint` = path to 336x600 checkpoint, `resume_only_model=true`
```bash
bash scripts/dist_train.sh 8 config_img +exp=448x800 runner=8gpus_img
```

**Debug Mode:**
To verify the environment and image data loading:

```bash
bash scripts/debug.sh config_img +exp=224x400 runner=debug runner.validation_before_run=true
```

### Test
We support two modes: `debug` (validation on a small subset) and `batch` (generation on the full validation set).

**Debug Mode (Subset Validation): Runs [tools/test.py](../tools/test.py)**
```bash
# bash scripts/dist_test_img.sh <NUM_GPUS> debug <CONFIG_NAME> <CKPT_PATH> [TASK_ID] [PY_ARGS]
bash scripts/dist_test_img.sh 1 debug test_config_img ${RESUME_CKPT} 224x400 +exp=224x400 runner=8gpus_img
```

**Batch Generation Mode (Full Validation Set): Runs [perception/data_prepare/val_set_gen.py](../perception/data_prepare/val_set_gen.py)**
```bash
# bash scripts/dist_test_img.sh <NUM_GPUS> batch <CONFIG_NAME> <CKPT_PATH> [TASK_ID] [PY_ARGS]
bash scripts/dist_test_img.sh 8 batch test_config_img ${RESUME_CKPT} 224x400 +exp=224x400 +fid=data_gen
```

## VideoGen Train & Test
### Train 

**Stage-1: Prepare ImgGen Pretrained Weights:**
Before starting the VideoGen training, you must ensure that the corresponding ImgGen diffusion weights are prepared and placed in the `pretrained` directory. These serve as the foundation for the video diffusion model.

Ensure your directory structure looks like this:
```bash
x-scene/pretrained/
├── x-scene-img_224x400/
├── x-scene-img_336x600/
└── x-scene-img_448x800/
```

**Stage-2: Train the Video Diffusion Model:**
We recommend a progressive training strategy, starting from low resolution and moving to higher resolutions.

* **Step 1: Low-Resolution Training ($224 \times 400$):**
Initialize the spatial layers with the `pretrained/x-scene-img_224x400` weights.
```bash
bash scripts/dist_train.sh 8 config_video +exp=224x400_t runner=8gpus_video
```

* **Step 2: Mid-Resolution Fine-tuning ($336 \times 600$):**
Update `config_video.yaml`: `resume_from_checkpoint` = path to 224x400_t checkpoint, `resume_only_model=true`. Ensure the model loads the `pretrained/x-scene-img_336x600` ImgGen weights for spatial features.
```bash
bash scripts/dist_train.sh 8 config_video +exp=336x600_t runner=8gpus_video
```

* **Step 3: High-Resolution Fine-tuning ($448 \times 800$):**
Update `config_video.yaml`: `resume_from_checkpoint` = path to 336x600_t checkpoint, `resume_only_model=true`. Ensure the model loads the `pretrained/x-scene-img_448x800` ImgGen weights for spatial features.
```bash
bash scripts/dist_train.sh 8 config_video +exp=448x800_t runner=8gpus_video
```

**Debug Mode:**
To verify the environment and video data loading:

```bash
bash scripts/debug.sh config_video +exp=224x400_t runner=debug runner.validation_before_run=true
```

### Test
We support two modes: `gen` (autoregressive video generation) and `submit` (16-frame videos or long videos generation for [w-coda2024/track2](https://coda-dataset.github.io/w-coda2024/track2/) submission). 

**Autoregressive Video Generation Mode: Runs [tools/test_video.py](../tools/test_video.py)**
```bash
# bash scripts/dist_test_video.sh <NUM_GPUS> gen [IMG_MODEL] [VIDEO_MODEL] [OUT_DIR]
bash scripts/dist_test_video.sh 8 gen pretrained/x-scene-img_224x400 pretrained/x-scene-video_224x400 work_dirs/test_video_gen
```

**Benchmark Submission Mode: Runs [tools/test_video_submit.py](../tools/test_video_submit.py)**
```bash
# bash scripts/dist_test_video.sh <NUM_GPUS> submit [IMG_MODEL] [VIDEO_MODEL] [OUT_DIR]
bash scripts/dist_test_video.sh 8 submit pretrained/x-scene-img_224x400 pretrained/x-scene-video_224x400 work_dirs/test_video_submit 
```

### Calculate Video FVD Score

1. **Environment Setup**  
   Prepare the Python environment and download the pre-trained weights as described in [tools/fvd_score/README.md](../tools/fvd_score/README.md).

2. **Run the Script**  
   Execute the base script with the following command:  

   ```bash
   bash scripts/cal_fvd_score.sh \
       work_dirs/test_video_submit \        # path to generated videos
       work_dirs/test_video_submit_fvd \    # path to save FVD results
   ```