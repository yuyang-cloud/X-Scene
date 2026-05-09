# paths
CKPT_PATH = 'ckpts'
CFG_PATH = 'confs'
WANDB_PATH = 'wandb_log'
OUT_PATH = 'out'

# train
TRAIN_HYDRA_ROOT = 'conf'
TRAIN_F2D_DEFAULT = 'train_f2d.yaml'
TRAIN_F3D_DEFAULT = 'train_f3d.yaml'

F2D_TRAIN_DEFAULT = TRAIN_F2D_DEFAULT
F3D_TRAIN_DEFAULT = TRAIN_F3D_DEFAULT

# generation
GEN_OCC_PATH = 'gen_occ'
GEN_IMG_PATH = 'gen_img'
GT_IMG_PATH = 'gt_img'

OCC_PATH = GEN_OCC_PATH
IMG_PATH = GT_IMG_PATH

# strings
CKPT_FILENAME_RULE = '{}_mIoU_{:.2f}.ckpt'
CKPT_LAST = 'last.ckpt'
CFG_FILENAME = 'config.yaml'
LOG_FILENAME = 'log.txt'
WANDB_PROJECT = 'DSC'

# others
EPSILON = 1e-10
