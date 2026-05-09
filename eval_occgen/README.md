# Eval_OccGen

This repository contains the occupancy evaluation code. It provides two learned feature extractors:

- **F3D**: a sparse 3D autoencoder trained directly on semantic occupancy grids.
- **F2D**: a 2D autoencoder trained on rendered occupancy images.

Both models are trained on ground-truth occupancy and then used as custom `torch-fidelity` feature extractors to evaluate generated occupancy results.

## Environment

Create a Python environment with PyTorch and CUDA support first. The code is tested with distributed training through `torchrun`.

Install the core dependencies:

```bash
pip install hydra-core omegaconf wandb tqdm numpy scipy numba pyquaternion pillow torchvision mmcv
```

Install the latest `torch-fidelity` version:

```bash
pip install -e git+https://github.com/toshas/torch-fidelity.git@master#egg=torch-fidelity
```

Install sparse convolution dependencies:

```bash
pip install torchsparse
```

Build the CUDA voxel rendering extension used for F2D image rendering:

```bash
cd eval_occgen/ext/voxlib
python setup.py build_ext --inplace
cd ../..
```

Optional but recommended:

```bash
wandb offline
```

The training scripts write checkpoints to `ckpts/`, cached configs to `confs/`, generated outputs to `out/`, and W&B logs to `wandb_log/`.

## Data Preparation

The default dataset config is `conf/dataset/occ3dn.yaml`.

Expected ground-truth Occ3D-nuScenes layout:

```text
eval_occgen/
  data/
    nuscenes/
      gts/
        <scene_name>/
          <sample_token>/
            labels.npz
      nuscenes_mmdet3d_2/
        nuscenes_infos_train.pkl
        nuscenes_infos_val.pkl
```

Generated occupancy for evaluation should be placed under:

```text
out/gen_occ/occ3dn/
  <sample_token>.npz
```

Each generated `.npz` should contain either:

- `occ`: semantic occupancy with shape `[200, 200, 16]`
- or `arr_0`: semantic occupancy with shape `[200, 200, 16]`

The generated file token must match the validation ground-truth token. For example, generated file `abc123.npz` is matched to ground truth folder `.../<scene_name>/abc123/labels.npz`.

## F3D Workflow

F3D is trained directly on ground-truth occupancy.

Train F3D:

```bash
cd eval_occgen
bash dist_train_f3d.sh 8
```

For a single GPU:

```bash
torchrun --nproc_per_node=1 --master_port=16842 train_f3d.py
```

Useful config overrides:

```bash
torchrun --nproc_per_node=1 train_f3d.py \
  name=F3D_occ3dn \
  dataset.batch_size=4 \
  trainer.num_epochs=100
```

Evaluate generated occupancy with the trained F3D checkpoint:

```bash
python eval_f3d.py -n F3D --sample-path out/gen_occ/occ3dn --best
```

By default, `eval_f3d.py` compares generated occupancy with the GT validation split. Use `--gt-split train` only if the generated files correspond to the training split.

## F2D Workflow

F2D first renders ground-truth occupancy into 2D images, trains a 2D autoencoder on those images, and then evaluates rendered generated results.

### Render Ground-Truth Occupancy Images

Build `ext/voxlib` before running this step.

Render the training split:

```bash
cd eval_occgen
python scripts/vis_img_cuda.py --dataset n --split train
```

Render the validation split:

```bash
python scripts/vis_img_cuda.py --dataset n --split val
```

The default output paths match `conf/dataset/occ3dn.yaml`:

```text
out/gt_img_train/occ3dn/
out/gt_img_val/occ3dn/
```

Each sample saves three rendered `.npy` views:

```text
<sample_token>_up_right.npy
<sample_token>_bev.npy
<sample_token>_multiview.npy
```

`train_f2d.py` trains on all `.npy` files in `trainset_path` and validates on all `.npy` files in `validset_path`.

### Train F2D

```bash
bash dist_train_f2d.sh 8
```

For a single GPU:

```bash
torchrun --nproc_per_node=1 --master_port=16842 train_f2d.py
```

Useful config overrides:

```bash
torchrun --nproc_per_node=1 train_f2d.py \
  name=F2D_occ3dn \
  dataset.batch_size=32 \
  trainer.num_epochs=100
```

### Prepare Rendered Generated Images

Render generated occupancy with the same camera setup and save `.npy` files under:

```text
out/gen_img/occ3dn/
```

The generated image tokens must match the validation ground-truth image tokens. For example:

```text
out/gt_img_val/occ3dn/abc123_bev.npy
out/gen_img/occ3dn/abc123_bev.npy
```

### Evaluate F2D

```bash
python eval_f2d.py -n F2D --sample-path out/gen_img/occ3dn --best
```

You can also explicitly pass the ground-truth image path:

```bash
python eval_f2d.py \
  -n F2D \
  --gt-path out/gt_img_val/occ3dn \
  --sample-path out/gen_img/occ3dn \
  --best
```

## Important Config Fields

`conf/dataset/occ3dn.yaml` controls the default paths:

```yaml
data_path: data/nuscenes
info_path: data/nuscenes/nuscenes_mmdet3d_2
trainset_path: out/gt_img_train/occ3dn
validset_path: out/gt_img_val/occ3dn
eval_gt_img_path: out/gt_img_val/occ3dn
eval_gen_img_path: out/gen_img/occ3dn
eval_gen_occ_path: out/gen_occ/occ3dn
num_classes: 18
grid_size: [200, 200, 16]
```

Use `conf/dataset/occ3dn_map.yaml` if you want the 12-class mapped version instead of the default 18-class labels:

```bash
torchrun --nproc_per_node=1 train_f3d.py dataset=occ3dn_map name=F3D_occ3dn_map
```

Keep the same dataset config when training and evaluating a checkpoint. A checkpoint trained with 18 classes should be evaluated with the 18-class config, and a checkpoint trained with 12 classes should be evaluated with the 12-class config.

## Output Files

Training produces:

```text
ckpts/<run_name>/last.ckpt
ckpts/<run_name>/<epoch>_mIoU_<score>.ckpt
confs/<run_name>/config.yaml
```

Evaluation uses `confs/<run_name>/config.yaml` to recover the training config and loads checkpoints from `ckpts/<run_name>/`.

## Notes

- All training entry points use distributed initialization. Use `torchrun` even for one GPU.
- Generated samples must be token-aligned with the ground-truth validation samples.
- F2D and F3D scores are not interchangeable. F2D evaluates rendered 2D occupancy images, while F3D evaluates the semantic 3D occupancy volume directly.
