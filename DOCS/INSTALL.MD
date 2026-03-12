# Installation

All the codes are tested in the following environment:

Linux (tested on Ubuntu 22.04)

CUDA 11.8

```bash
# Install a virtual environment.
conda create -n xscene python=3.9
conda activate xscene
```

**1. Install `torch==2.0.0+cu118`**
```bash
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
```

**2. Install the dependencies**
```bash
pip install -r requirements.txt
```

**3. Install mm-series**
```bash
pip install mmcv-full==1.6.0
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0
```

**4. Install the source code**

```
# Install third-party packages
third_party/
├── mmdetection3d
└── diffusers -> based on v0.17.1 (afcca3916)

cd third_party/mmdetection3d
pip install -v -e .
cd third_party/diffuser
pip install -v -e .
```