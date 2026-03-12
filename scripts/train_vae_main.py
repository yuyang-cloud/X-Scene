import argparse
import torch
import torch.distributed as dist
from xscene.occ_vae.train_ae import Trainer

# Occ3D-nuScenes    range=[-40m,-40m,-1m,40m,40m,5.4m]       voxel_size=[0.4m,0.4m,0.4m] volume_size=[200,200,16]
NUSCENES_OCC3D_DATA_PATH = 'data/nuscenes/gts'
NUSCENES_OCC3D_TRAIN_INFO_PATH = 'data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl'
NUSCENES_OCC3D_VAL_INFO_PATH = 'data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl'
# nuScene_Ocucpancy range=[-51.2m,-51.2m,-5m,51.2m,51.2m,3m] voxel_size=[0.2m,0.2m,0.2m] volume_size=[512,512,40]
NUSCENES_OCCUPANCY_DATA_PATH = 'data/nuscenes/nuScenes-Occupancy'
NUSCENES_OCCUPANCY_TRAIN_INFO_PATH = 'data/nuscenes/nuscenes_infos_temporal_train_new.pkl'
NUSCENES_OCCUPANCY_VAL_INFO_PATH = 'data/nuscenes/nuscenes_infos_temporal_val_new.pkl'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo_feat_channels", type=int, default=16, help="geometry feature dimension")
    parser.add_argument("--feat_channel_up", type=int, default=64, help="conv feature dimension")
    parser.add_argument("--mlp_hidden_channels", type=int, default=256, help="mlp hidden dimension")
    parser.add_argument("--mlp_hidden_layers", type=int, default=4, help="mlp hidden layers")
    parser.add_argument("--padding_mode", default='replicate')
    parser.add_argument("--bs", type=int, default=24, help="batch size for autoencoding training")
    parser.add_argument("--dataset", default='Occ3D-nuScenes', choices=['Occ3D-nuScenes', 'nuScenes-Occupancy'])
    parser.add_argument("--z_down", action='store_true')
    parser.add_argument("--xy_down", type=bool, default=True)
    parser.add_argument("--xy_down_times", type=int, default=2, choices=[2, 4], help="downsample times for triplane_xy")
    parser.add_argument("--use_vae", action='store_true', help="use VAE")

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--steplr_scheduler", default=False)
    parser.add_argument("--lr_scheduler_steps", nargs='+', type=int, default=[40, 80])
    parser.add_argument("--lr_scheduler_decay", type=float, default=0.1)
    parser.add_argument("--global_seed", type=int, default=1)

    parser.add_argument('--save_path', type=str, default='work_dirs/vae')
    parser.add_argument('--resume', default = None)
    parser.add_argument('--display_period', type=int, default=1)
    parser.add_argument("--total_epoch", type=int, default=200)
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--eval_during_training', type=bool, default=False)
    
    parser.add_argument("--triplane", type=bool, default=True, help="use triplane feature, if False, use bev feature")
    parser.add_argument("--use_deform_attn", action='store_true', help="use deformable attention for triplane feature")
    parser.add_argument("--pos", default=True, type=bool)
    parser.add_argument("--voxel_fea", default=False, type=bool, help="use 3d voxel feature")
    args = parser.parse_args()
    return args

def main(args, rank=0):
    # init DDP
    if torch.cuda.device_count() > 1:
        distributed = True
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        args.lr *= torch.cuda.device_count()
        args.world_size = dist.get_world_size()
        print(f"Starting rank={rank}, world_size={args.world_size}.")
    else:
        distributed = False
        args.world_size = 1

    torch.cuda.set_device(rank)

    trainer = Trainer(args, rank, distributed)
    trainer.train()

if __name__ == '__main__':
    args = get_args()

    if args.dataset == 'Occ3D-nuScenes':
        args.data_path=NUSCENES_OCC3D_DATA_PATH
        args.train_info_path=NUSCENES_OCC3D_TRAIN_INFO_PATH
        args.val_info_path=NUSCENES_OCC3D_VAL_INFO_PATH
    elif args.dataset == 'nuScenes-Occupancy':
        args.data_path=NUSCENES_OCCUPANCY_DATA_PATH
        args.train_info_path=NUSCENES_OCCUPANCY_TRAIN_INFO_PATH
        args.val_info_path=NUSCENES_OCCUPANCY_VAL_INFO_PATH
    else:
        raise NotImplementedError

    main(args)
