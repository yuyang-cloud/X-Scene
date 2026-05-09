import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyrender
import tqdm
import yaml
from PIL import Image

import occgen.utils.constants as C
from scripts.vis_utils import get_mesh_open3d, get_mesh_trimesh, parse_xml
from scripts.xmls import XMLS


def render_scene_offscreen(mesh, output_path, camera, camera_pose, light):
    scene = pyrender.Scene()

    scene.add(mesh)
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    try:
        r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)
    except AttributeError:
        return True

    color, depth = r.render(scene)
    imageio.imwrite(output_path, color)
    r.delete()
    return False


def render_video(out_paths, camera, camera_pose, light, im_save_pattern=None):
    images = []
    for i, mesh in enumerate(tqdm.tqdm(meshes)):
        output_path = f'temp_frame_{i:04d}_{args.name}.png'

        if args.outpaint:
            camera_out, camera_pose_out = parse_xml(XMLS['outpaintc'])
            while render_scene_offscreen(mesh, output_path, camera_out, camera_pose_out, light): pass
        else:
            while render_scene_offscreen(mesh, output_path, camera, camera_pose, light): pass

        if args.compare:
            output_orig_path = f'temp_frame_orig_{i:04d}_{args.name}.png'
            while render_scene_offscreen(meshes_orig[i], output_orig_path, camera, camera_pose, light): pass

            img_pred = imageio.imread(output_path)
            img_orig = imageio.imread(output_orig_path)

            imageio.imwrite(im_save_pattern.format(f'{i}_pred'), img_pred)
            imageio.imwrite(im_save_pattern.format(f'{i}_orig'), img_orig)

            combined_img = np.hstack((img_orig, img_pred))
            images.append(combined_img)

            os.remove(output_path)
            os.remove(output_orig_path)
        elif args.layout:
            img_pred = imageio.imread(output_path)  # H, W, 3
            img_layout = imageio.imread(layout_images[i])  # X, X, 4

            img_layout = Image.fromarray(img_layout[:, :, :3])
            img_layout = img_layout.resize((img_pred.shape[1], img_pred.shape[0]))

            img_layout = np.array(img_layout)
            # imageio.imwrite(im_save_pattern.format(f'{i}_layout'), img_layout)
            # imageio.imwrite(im_save_pattern.format(f'{i}_pred'), img_pred)
            combined_img = np.concatenate((img_layout, img_pred), axis=1)

            images.append(combined_img)
            os.remove(output_path)
        else:
            images.append(imageio.imread(output_path))
            os.remove(output_path)

    for path in out_paths:
        if str(path).endswith(".gif"):
            imageio.mimwrite(path, images, fps=FPS, loop=0)
        else:
            imageio.mimwrite(path, images, fps=FPS)

    if im_save_pattern is not None:
        for i, image in enumerate(images):
            imageio.imwrite(im_save_pattern.format(i), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-d', '--dataset', type=str, choices=['c', 'n', 'w'], required=True)
    parser.add_argument('--ignore', type=int, default=0)
    parser.add_argument('--xml', type=str, default='default')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--vox', action='store_true')
    parser.add_argument('--o3d', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--layout', action='store_true')
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--outpaint', action='store_true')
    parser.add_argument('--single', action='store_true')
    args = parser.parse_args()

    VOXEL_SIZE = 1.0
    DATASET = ['carlasc', 'occ3dn', 'occ3dw']['cnw'.index(args.dataset)]
    FPS = [5, 2, 5]['cnw'.index(args.dataset)] if args.fps is None else args.fps
    CFG_PATH = f'conf/vae/dataset/{DATASET}.yaml'
    XML = XMLS[args.xml]

    ignores, postfix = None, ''
    # ignores, postfix = [3, 4, 10, 11], '_static'
    # ignores, postfix = [1, 2, 3, 5, 6, 7, 8, 9], '_dynamic'

    cfg_file = yaml.safe_load(open(CFG_PATH, 'r'))
    COLOR_MAP = cfg_file['color_map']  # BGR
    SHAPE = cfg_file['grid_size']

    target_root = Path(C.RECONSTRUCT_PATH) / ((C.GEN_PATH if args.gen else C.VOXEL_PATH) + '') / args.name
    gif_root = Path(C.RECONSTRUCT_PATH) / C.GIF_PATH / args.name
    video_root = Path(C.RECONSTRUCT_PATH) / C.VIDEO_PATH / args.name
    webp_root = Path(C.RECONSTRUCT_PATH) / 'webp' / args.name
    im_root = Path(C.RECONSTRUCT_PATH) / C.IMAGE_PATH / args.name

    gif_root.mkdir(exist_ok=True, parents=True)
    video_root.mkdir(exist_ok=True, parents=True)
    webp_root.mkdir(exist_ok=True, parents=True)
    im_root.mkdir(exist_ok=True, parents=True)

    target_folders = sorted(
        target_root.iterdir(), key=lambda x: int(''.join(filter(str.isdigit, x.name))) if
        any(c.isdigit() for c in x.name) else 0
    )
    target_folders = target_folders[args.start: min(len(target_folders), args.limit)]

    for target_folder in target_folders:
        if '.DS_Store' in str(target_folder):
            continue

        # if 'train' in str(target_folder):
        #     continue

        if 'layout' in str(target_folder):
            continue

        # selected = [29, 43, 58, 73, 87, 121]
        # selected = [14, 29, 31, 41]
        # selected = [37, 53]
        selected = [16]
        found = False
        for select in selected:
            if str(select) in str(target_folder)[-4:]:
                found = True
        if not found:
            print(str(target_folder)[-4:])
            continue
        #
        # if '07' in str(target_folder):
        #     continue

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

        get_mesh = get_mesh_open3d if args.o3d else get_mesh_trimesh

        # frames = range(1080, 1260, 1)
        frames = range(0, T)

        # frames = [0, 3, 6, 9, 12, 15]
        # frames = [29]
        # frames = [0, 15]
        # if len(samples) < 16:
        #     continue
        mask = None
        # mask = [30, 60, 60, 70]
        # mask = [0, 128, 50, 70]
        # mask = [0, 128, 0, 70]
        meshes = list()
        meshes_orig = list()

        vox_shape = SHAPE.copy()
        if args.outpaint:
            vox_shape[0] = SHAPE[0] * 2
        voxels = [np.fromfile(sample_path, dtype=np.int8).reshape(vox_shape) for sample_path in samples]

        if args.compare:
            if args.outpaint:
                voxels_orig = [np.fromfile(sample_path, dtype=np.uint8).reshape(SHAPE) for sample_path in samples_orig]
                for i in range(len(voxels_orig)):
                    vox = np.zeros((SHAPE[0] * 2, SHAPE[1], SHAPE[2]), dtype=np.uint8)
                    vox[64: 192] = voxels_orig[i][:]
                    voxels_orig[i] = vox
            else:
                voxels_orig = [np.fromfile(sample_path, dtype=np.uint8).reshape(SHAPE) for sample_path in samples_orig]

        try:
            for i in tqdm.tqdm(frames):
                vox = voxels[i]
                if args.compare and mask is not None:
                    vox = voxels_orig[i].copy()
                    # vox[mask[0]: mask[1], mask[2]: mask[3]] = 0
                    vox[mask[0]: mask[1], mask[2]: mask[3]] = voxels[i][mask[0]: mask[1], mask[2]: mask[3]]
                # if i == 0:
                #     get_mesh(vox, vox_shape, args.ignore, COLOR_MAP, VOXEL_SIZE, ignores)[0].export('mesh.ply')
                meshes.append(get_mesh(vox, vox_shape, args.ignore, COLOR_MAP, VOXEL_SIZE, ignores)[1])
                if args.compare:
                    if args.outpaint:
                        meshes_orig.append(get_mesh(voxels_orig[i], vox_shape, args.ignore, COLOR_MAP, VOXEL_SIZE, ignores)[1])
                    elif args.single:
                        meshes_orig.append(
                            get_mesh(voxels_orig[-1], SHAPE, args.ignore, COLOR_MAP, VOXEL_SIZE, ignores)[1])
                    else:
                        meshes_orig.append(get_mesh(voxels_orig[i], SHAPE, args.ignore, COLOR_MAP, VOXEL_SIZE, ignores)[1])
        except Exception as e:
            print(e)
            continue

        if args.layout:
            layout_images = list(map(str, sorted(target_folder.glob('*.png'), key=lambda x: int(str(x.stem)))))

        camera, camera_pose = parse_xml(XML)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)

        out_paths = [
            gif_root / f'{target_folder.name}{postfix}.gif',
            video_root / f'{target_folder.name}{postfix}.mp4',
            webp_root / f'{target_folder.name}{postfix}.webp'
        ]
        im_save_pattern = str(im_root / (target_folder.name + '_{}.png'))
        render_video(out_paths, camera, camera_pose, light, im_save_pattern)

        print(f'Video {out_paths[1]} saved to {out_paths[1].parent}.')
        print()
