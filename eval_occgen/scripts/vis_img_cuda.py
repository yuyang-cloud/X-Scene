import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import occgen.utils.constants as C
from occgen.dataset.carlasc import CarlaSCVoxel
from occgen.dataset.f3d import F3DDataset
from occgen.dataset.occ3dn import Occ3DNVoxel
from occgen.dataset.occ3dw import Occ3DWVoxel
from occgen.dataset.semkitti import SemKittiVoxel
from ext.voxlib import ray_voxel_intersection_perspective
from concurrent.futures import ThreadPoolExecutor
from PIL import Image


def render_voxel(voxel, colormap, params):
    out = ray_voxel_intersection_perspective(voxel, *params)[0][:, :, 0, 0]
    return colormap[out].cpu().numpy().astype(np.uint8)


def render_and_merge(voxel, camera_params):
    images = []
    for i, params in enumerate(camera_params):
        image = render_voxel(voxel, colormap, params)
        vis_img = Image.fromarray(image)
        images.append(vis_img)

    # concate front_views
    first_row = Image.new('RGB', (images[0].width + images[1].width + images[2].width, images[0].height))
    first_row.paste(images[0], (0, 0))
    first_row.paste(images[1], (images[0].width, 0))
    first_row.paste(images[2], (images[0].width + images[1].width, 0))

    # concate back_views
    second_row = Image.new('RGB', (images[3].width + images[4].width + images[5].width, images[3].height))
    second_row.paste(images[3], (0, 0))
    second_row.paste(images[4], (images[3].width, 0))
    second_row.paste(images[5], (images[3].width + images[4].width, 0))

    # concate front and back views
    final_image = Image.new('RGB', (first_row.width, first_row.height + second_row.height))
    final_image.paste(first_row, (0, 0))
    final_image.paste(second_row, (0, first_row.height))

    return np.array(final_image)

def main(up_right_params, bev_params, camera_params, vis=True):
    dataset_class = {
        'carlasc': CarlaSCVoxel,
        'occ3dn': Occ3DNVoxel,
        'occ3dw': Occ3DWVoxel,
        'semkitti': SemKittiVoxel,
    }[DATASET]
    dataset = dataset_class(cfg_file, imageset=args.split)

    if args.save_dataset:
        for i, data in tqdm(enumerate(dataset), total=len(dataset)):
            voxel = torch.tensor(data['voxel']).to(torch.int32).cuda()
            up_right_image = render_voxel(voxel, colormap, up_right_params)
            bev_image = render_voxel(voxel, colormap, bev_params)
            multiview_img = render_and_merge(voxel, camera_params)

            token = Path(data['path']).parent.name

            np.save(dataset_save_root / f'{token}_up_right.npy', up_right_image)
            np.save(dataset_save_root / f'{token}_bev.npy', bev_image)
            np.save(dataset_save_root / f'{token}_multiview.npy', multiview_img)

            if vis:
                vis_img = Image.fromarray(up_right_image)
                vis_img.save(vis_save_root / f'{token}_up_right.jpg')

                bev_img = Image.fromarray(bev_image)
                bev_img.save(vis_save_root / f'{token}_bev.jpg')

                multiview_img = Image.fromarray(multiview_img)
                multiview_img.save(vis_save_root / f'{token}_camera.jpg')



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['c', 'n', 'w', 'k'], default='n')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='val')
    parser.add_argument('--output-root', type=str, default=None)
    parser.add_argument('--save_dataset', action='store_true', default=True)
    parser.add_argument('--vis', action='store_true', default=True, help='Whether to save visualization images')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    VOXEL_SIZE = 1.0
    DATASET = ['carlasc', 'occ3dn', 'occ3dw', 'semkitti']['cnwk'.index(args.dataset)]
    CFG_PATH = ROOT / 'conf' / 'dataset' / f'{DATASET}.yaml'

    cfg_file = OmegaConf.load(CFG_PATH)
    COLOR_MAP = cfg_file['color_map']  # BGR
    SHAPE = cfg_file['grid_size']

    colormap = list(COLOR_MAP.values())
    colormap = torch.tensor(colormap, dtype=torch.float32).cuda()

    if args.dataset == 'n':
        learning_map = torch.tensor(list(cfg_file['learning_map'].values())).cuda()
    elif args.dataset == 'k':
        learning_map_dict = cfg_file['learning_map']
        max_key = max(learning_map_dict.keys())
        learning_map = torch.zeros(max_key + 1, dtype=torch.int64)
        for k, v in learning_map_dict.items():
            learning_map[k] = v
    else:
        learning_map = None

    if args.output_root is None:
        split_name = 'train' if args.split == 'train' else 'val'
        dataset_save_root = Path(C.OUT_PATH) / f'{C.GT_IMG_PATH}_{split_name}' / DATASET
    else:
        dataset_save_root = Path(args.output_root)
    dataset_save_root.mkdir(exist_ok=True, parents=True)
    vis_save_root = dataset_save_root / "vis_img"
    vis_save_root.mkdir(exist_ok=True, parents=True)

    # up-right view
    XYZ_FACTOR = [-2, -2, 25]
    IMAGE_DIMS = [400, 800]
    cam_ori = torch.tensor([dim * factor for dim, factor in zip(SHAPE, XYZ_FACTOR)], dtype=torch.float32)
    cam_dir = torch.tensor([SHAPE[0] // 2, SHAPE[1] // 2, 0], dtype=torch.float32) - cam_ori
    cam_up = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    cam_f = max(IMAGE_DIMS) / 0.35
    cam_c = [dim // 2 for dim in IMAGE_DIMS]
    img_dims = IMAGE_DIMS
    max_samples = 1
    up_right_params = [cam_ori, cam_dir, cam_up, cam_f, cam_c, img_dims, max_samples]

    # BEV view
    XYZ_FACTOR = [0, 0, 35]
    IMAGE_DIMS = [400, 400]
    cam_ori = torch.tensor([SHAPE[0] / 2, SHAPE[1] / 2, SHAPE[2] * XYZ_FACTOR[2]], dtype=torch.float32)
    cam_dir = torch.tensor([0, 0, -1], dtype=torch.float32)
    cam_up = torch.tensor([0, 1, 0], dtype=torch.float32)
    cam_f = max(IMAGE_DIMS) / 0.35
    cam_c = [dim // 2 for dim in IMAGE_DIMS]
    img_dims = IMAGE_DIMS
    max_samples = 1
    bev_params = [cam_ori, cam_dir, cam_up, cam_f, cam_c, img_dims, max_samples]

    # multi-view camera
    camera_configs = [
        {"name": "FRONT_LEFT",  "cam_ori": [2, 2, 1.5], "cam_dir": [1, 1, 0]},
        {"name": "FRONT",       "cam_ori": [2, 0, 1.5], "cam_dir": [1, 0, 0]},
        {"name": "FRONT_RIGHT", "cam_ori": [2, -2, 1.5], "cam_dir": [1, -1, 0]},
        {"name": "BACK_RIGHT",  "cam_ori": [-2, -2, 1.5], "cam_dir": [-1, -1, 0]},
        {"name": "BACK",        "cam_ori": [-2, 0, 1.5], "cam_dir": [-1, 0, 0]},
        {"name": "BACK_LEFT",   "cam_ori": [-2, 2, 1.5], "cam_dir": [-1, 1, 0]},
    ]
    IMAGE_DIMS = [200, 400]
    cam_up = torch.tensor([0, 0, 1], dtype=torch.float32)
    cam_f = max(IMAGE_DIMS) / 0.91
    cam_c = [dim // 2 for dim in IMAGE_DIMS]
    img_dims = IMAGE_DIMS
    max_samples = 1
    camera_params = []
    for config in camera_configs:
        cam_ori = torch.tensor(config["cam_ori"] + np.array([SHAPE[0]/2, SHAPE[1]/2, 5]), dtype=torch.float32)
        cam_dir = torch.tensor(config["cam_dir"], dtype=torch.float32)
        params = [cam_ori, cam_dir, cam_up, cam_f, cam_c, img_dims, max_samples]
        camera_params.append(params)

    main(up_right_params, bev_params, camera_params)
    
