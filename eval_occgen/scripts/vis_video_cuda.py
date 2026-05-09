import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm

import occgen.utils.constants as C
from ext.voxlib import ray_voxel_intersection_perspective


def render_voxel(voxel, colormap, params):
    out = ray_voxel_intersection_perspective(voxel, *params)[0][:, :, 0, 0]
    return colormap[out].cpu().numpy().astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-d', '--dataset', type=str, choices=['c', 'n', 'w'], required=True)
    parser.add_argument('--ignore', type=int, default=0)

    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--vox', action='store_true')

    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--out', action='store_true')
    args = parser.parse_args()

    VOXEL_SIZE = 1.0
    DATASET = ['carlasc', 'occ3dn', 'occ3dw']['cnw'.index(args.dataset)]
    FPS = [10, 2, 10]['cnw'.index(args.dataset)] if args.fps is None else args.fps
    CFG_PATH = f'conf/vae/dataset/{DATASET}.yaml'

    cfg_file = yaml.safe_load(open(CFG_PATH, 'r'))
    COLOR_MAP = cfg_file['color_map']  # BGR
    SHAPE = cfg_file['grid_size']

    colormap = list(COLOR_MAP.values())
    colormap = torch.tensor(colormap, dtype=torch.float32).cuda()
    colormap = colormap[:, [2, 1, 0]]

    target_root = Path(C.RECONSTRUCT_PATH) / (C.GEN_PATH if args.gen else C.VOXEL_PATH) / args.name
    gif_root = Path(C.RECONSTRUCT_PATH) / C.GIF_PATH / args.name
    video_root = Path(C.RECONSTRUCT_PATH) / C.VIDEO_PATH / args.name

    XYZ_FACTOR = [-2, -2, 25]
    IMAGE_DIMS = [1024, 1024]

    cam_ori = torch.tensor([dim * factor for dim, factor in zip(SHAPE, XYZ_FACTOR)], dtype=torch.float32)
    cam_dir = torch.tensor([SHAPE[0] // 2, SHAPE[1] // 2, 0], dtype=torch.float32) - cam_ori
    cam_up = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    cam_f = max(IMAGE_DIMS) / 0.35
    cam_c = [dim // 2 for dim in IMAGE_DIMS]
    img_dims = IMAGE_DIMS
    max_samples = 1
    params = [cam_ori, cam_dir, cam_up, cam_f, cam_c, img_dims, max_samples]

    gif_root.mkdir(exist_ok=True, parents=True)
    video_root.mkdir(exist_ok=True, parents=True)

    for target_folder in sorted(target_root.iterdir()):
        if 'train' in str(target_folder):
            continue

        print(f'Visualizing {target_folder}')
        if args.compare:
            pattern = '*_pred.label'
            samples_orig = list(
                map(
                    str, sorted(
                        target_folder.glob('*_orig.label'),
                        key=lambda x: int(str(x.stem).split('_')[0])
                    )
                )
            )
        elif args.original:
            pattern = '*_orig.label'
        else:
            pattern = '*.label' if args.gen else '*[!_orig].label'

        samples = list(map(str, sorted(target_folder.glob(pattern), key=lambda x: int(str(x.stem).split('_')[0]))))
        T = len(samples)

        voxels_shape = SHAPE.copy()
        if args.out:
            voxels_shape[0] = SHAPE[0] * 2
        voxels = [torch.from_numpy(np.fromfile(sample, dtype=np.uint8).reshape(voxels_shape)).cuda().to(torch.int32) for sample
            in tqdm(samples)]
        images = [render_voxel(voxel, colormap, params) for voxel in voxels]

        if args.compare:
            voxels_orig = [torch.from_numpy(np.fromfile(sample, dtype=np.uint8).reshape(SHAPE)).cuda().to(torch.int32)
                for sample in tqdm(samples_orig)]
            images_orig = [render_voxel(voxel, colormap, params) for voxel in voxels_orig]
            images = [np.hstack([image_orig, image]) for image_orig, image in zip(images_orig, images)]

        out_paths = [
            video_root / f'{target_folder.name}.mp4'
        ]

        [imageio.mimwrite(out_path, images, fps=FPS) for out_path in out_paths]

        print(f'Video {out_paths[0]} saved to {out_paths[0].parent}.')
        print()
