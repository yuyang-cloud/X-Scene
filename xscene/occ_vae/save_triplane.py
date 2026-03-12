import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)
import torch
import numpy as np
import argparse
from xscene.occ_vae.dataset.nuscenes_occ_dataset import NuScenesDatasetOcc 
from xscene.occ_vae.networks.networks import AutoEncoderGroupSkip
from xscene.occ_vae.utils.utils import compose_featmaps, analyze_latent_distributions
from tqdm.auto import tqdm
from pathlib import Path

# Occ3D-nuScenes    range=[-40m,-40m,-1m,40m,40m,5.4m]       voxel_size=[0.4m,0.4m,0.4m] volume_size=[200,200,16]
NUSCENES_OCC3D_DATA_PATH = 'data/nuscenes/gts'
NUSCENES_OCC3D_TRAIN_INFO_PATH = 'data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl'
NUSCENES_OCC3D_VAL_INFO_PATH = 'data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl'
# nuScene_Ocucpancy range=[-51.2m,-51.2m,-5m,51.2m,51.2m,3m] voxel_size=[0.2m,0.2m,0.2m] volume_size=[512,512,40]
NUSCENES_OCCUPANCY_DATA_PATH = 'data/nuscenes/nuScenes-Occupancy'
NUSCENES_OCCUPANCY_TRAIN_INFO_PATH = 'data/nuscenes/nuscenes_infos_temporal_train_new.pkl'
NUSCENES_OCCUPANCY_VAL_INFO_PATH = 'data/nuscenes/nuscenes_infos_temporal_val_new.pkl'
# Triplane
NUSCENES_TRIPLANE_PATH = 'data/nuscenes/nuscenes_triplane'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo_feat_channels", type=int, default=16, help="geometry feature dimension")
    parser.add_argument("--feat_channel_up", type=int, default=64, help="conv feature dimension")
    parser.add_argument("--mlp_hidden_channels", type=int, default=256, help="mlp hidden dimension")
    parser.add_argument("--mlp_hidden_layers", type=int, default=4, help="mlp hidden layers")
    parser.add_argument("--z_down", action='store_true', help="downsample in z-axis for triplane_z")
    parser.add_argument("--xy_down", type=bool, default=True)
    parser.add_argument("--xy_down_times", type=int, default=2, choices=[2, 4], help="downsample times for triplane_xy")
    parser.add_argument("--padding_mode", default='replicate')
    parser.add_argument('--lovasz', type=bool, default=True)
    parser.add_argument("--use_vae", action='store_true', help="use VAE")
    parser.add_argument("--use_deform_attn", action='store_true', help="use deformable attention for triplane feature")

    parser.add_argument("--dataset", default='Occ3D-nuScenes', choices=['Occ3D-nuScenes', 'nuScenes-Occupancy'])
    parser.add_argument('--data_name', default='voxels')
    parser.add_argument('--data_tail', default='.label')
    parser.add_argument('--save_name', default='triplane')
    parser.add_argument('--save_tail', default='.npy')
    parser.add_argument('--resume', default = 'work_dirs/vae/best_model')
    
    parser.add_argument("--triplane", type=bool, default=True)
    parser.add_argument("--pos", default=True, type=bool)
    parser.add_argument("--voxel_fea", default=False, type=bool)
    args = parser.parse_args()
    return args

@torch.no_grad()
def save(args):    
    if args.dataset == 'Occ3D-nuScenes':
        dataset = NuScenesDatasetOcc(args, 'train', get_query=False)
        val_dataset = NuScenesDatasetOcc(args, 'val', get_query=False)
        if args.xy_down:
            if args.xy_down_times == 2:
                x_size, y_size = 100, 100
            elif args.xy_down_times == 4:
                x_size, y_size = 50, 50
        else:
            x_size, y_size = 200, 200
        if args.z_down:
            z_size = 8
        else:
            z_size = 16
        tri_size = (x_size, y_size, z_size)
    elif args.dataset == "nuScenes-Occupancy":
        dataset = NuScenesDatasetOcc(args, 'train', get_query=False)
        val_dataset = NuScenesDatasetOcc(args, 'val', get_query=False)
        tri_size = (256, 256, 20) if args.z_down else (256, 256, 40)
    else:
        raise ValueError(f"dataset {args.dataset} not supported")
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)  #collate_fn=dataset.collate_fn, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)  #collate_fn=dataset.collate_fn, num_workers=4)
    
    print(args.data_name)
    print(f'The number of voxel labels is {len(dataset)}.')
    print(f'Load autoencoder model from "{args.resume}"')
    model = AutoEncoderGroupSkip.from_pretrained(args.resume)
    model = model.cuda()
    model.eval()

    print('\nAnalyze latent distributions...')
    analyze_latent_distributions(model, val_dataloader, device='cuda')

    print("\nSave Triplane...")
    for loader in [dataloader, val_dataloader]:
        for vox, _, _, _, path, invalid in tqdm(loader):
            # to gpu
            vox = vox.type(torch.LongTensor).cuda()         # [256,256,3] vox_label
            invalid = invalid.type(torch.LongTensor).cuda() # [256,256,3]
            vox[invalid == 1] = 0
            output = model.encode(vox)                    # [B,C,X,Y  B,C,X,Z  B,C,Y,Z]
            
            if not args.voxel_fea :
                if args.use_vae:
                    triplane = output[0]
                else:
                    triplane = output
                triplane, _ = compose_featmaps(triplane[0].squeeze(), triplane[1].squeeze(), triplane[2].squeeze(), tri_size)   # B,C,X+Y,Y+Z

            if args.dataset == "Occ3D-nuScenes":
                frame_token = path[0].split('/')[-2]    # frame_token
                scene_token = path[0].split('/')[-3]     # scene_name
            elif args.dataset == "nuScenes-Occupancy":
                frame_token = path[0].split('/')[-1].split('.')[0]  # lidar_token
                scene_token = path[0].split('/')[-3]                # scene_token
            save_folder_path = os.path.join(args.save_path, scene_token, args.save_name)
            os.makedirs(save_folder_path, exist_ok=True)
            np.save(os.path.join(save_folder_path, frame_token +args.save_tail), triplane.cpu().numpy())   
        
def main():
    args = get_args()
    if args.dataset == 'Occ3D-nuScenes':
        args.num_class = 18
        args.grid_size = [200, 200, 16]
        args.pc_range = [-40, -40, -1, 40, 40, 5.4]
        args.data_path=NUSCENES_OCC3D_DATA_PATH
        args.save_path=NUSCENES_TRIPLANE_PATH
        args.train_info_path=NUSCENES_OCC3D_TRAIN_INFO_PATH
        args.val_info_path=NUSCENES_OCC3D_VAL_INFO_PATH
    elif args.dataset == 'nuScenes-Occupancy':
        args.num_class = 18
        args.grid_size = [512, 512, 40]
        args.pc_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
        args.data_path=NUSCENES_OCCUPANCY_DATA_PATH
        args.save_path=NUSCENES_TRIPLANE_PATH
        args.train_info_path=NUSCENES_OCCUPANCY_TRAIN_INFO_PATH
        args.val_info_path=NUSCENES_OCCUPANCY_VAL_INFO_PATH
    else:
        raise ValueError(f"dataset {args.dataset} not supported")

    save(args)
    
if __name__ == '__main__':
    main()
