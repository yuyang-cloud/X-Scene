import os
import sys
import hydra
from hydra import initialize, compose
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
from PIL import ImageOps, Image
from moviepy.editor import *
import random
import time

import torch
import torchvision
from mmdet3d.datasets import build_dataset
from argparse import ArgumentParser

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

sys.path.append(".")  # noqa
from xscene.runner.utils import concat_6_views, img_concat_h, img_concat_v
from xscene.misc.test_utils import (
    build_pipe, run_one_batch_img, run_one_batch_video, update_progress_bar_config, collate_fn, ListSetWrapper, partial
)

transparent_bg = False
target_map_size = 400
# target_map_size = 800

# def output_func(x): return concat_6_views(x)

class ImageNormalize: ## Important !!! check the mean and std which should be consistent with the dataset.
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.compose(img)


def load_data(cfg, pipe):
    #### datasets ####
    val_param = OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
    val_dataset = build_dataset(val_param)

    #### dataloader ####
    if cfg.runner.validation_index != "all":
        val_dataset = ListSetWrapper(val_dataset, cfg.runner.validation_index)

    if hasattr(cfg.model, "ref_length"):
        assert cfg.runner.validation_batch_size == 1, "Do not support more."
    collate_fn_param = {
        "tokenizer_clip": pipe.tokenizer if hasattr(pipe, "tokenizer_t5") else None,
        "tokenizer_t5": pipe.tokenizer_t5 if hasattr(pipe, "tokenizer_t5") else None,
        "template_clip": cfg.dataset.template_clip,
        "template_t5": cfg.dataset.template_t5,
        "bbox_mode": cfg.model.bbox_mode,
        "bbox_view_shared": cfg.model.bbox_view_shared,
        "bbox_drop_ratio": cfg.runner.bbox_drop_ratio,
        "bbox_add_ratio": cfg.runner.bbox_add_ratio,
        "bbox_add_num": cfg.runner.bbox_add_num,
    }

    if hasattr(cfg.model, "ref_length"):
        collate_fn_param['ref_length'] = cfg.model.ref_length

    def _collate_fn(examples, *args, **kwargs):
        if hasattr(cfg.model, "ref_length"):
            return collate_fn(examples[0], *args, **kwargs)
        else:
            return collate_fn(examples, *args, **kwargs)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=partial(_collate_fn, is_train=False, **collate_fn_param),
        batch_size=cfg.runner.validation_batch_size,
        num_workers=cfg.runner.num_workers,
    )
    return val_dataloader


def output_func(x): return concat_6_views(x, oneline=True)
# def output_func(x): return img_concat_h(*x[:3])


def make_video_with_filenames(filenames, outname, fps=2):
    clips = [ImageClip(m).set_duration(1 / fps) for m in filenames]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(outname, fps=fps)


def generate_ref_with_single_pipe(val_input, weight_dtype):
    keys = ['meta_data', 'captions_clip', 'captions_t5', "pixel_values", "camera_param", 'kwargs', "bev_map_with_aux", "bev_hdmap", "layout_canvas"]
    val_input_single = {}
    for key in keys:
        if key == 'meta_data':
            val_input_single[key] = {'metas': val_input[key]['metas'][:1], 'lidar2image': val_input[key]['lidar2image'][:1], 'gt_bboxes_3d': val_input[key]['gt_bboxes_3d'][:1]}
        elif key == 'kwargs':
            val_input_single[key] = {'bboxes_3d_data': {k:v[:1] for k,v in val_input[key]['bboxes_3d_data'].items()}}
        else:
            val_input_single[key] = val_input[key][:1]
    cfg_single.runner.validation_times = 1
    cfg_single.show_box = False
    cfg_single.show_lane = False
    gen_imgs_list = run_one_batch_img(cfg_single, pipe_single, val_input_single, weight_dtype)[4]
    
    ref_image = torch.stack([ImageNormalize()(x) for x in gen_imgs_list[0][0]])
    ref_images = torch.stack([ref_image, ref_image.clone()])
    val_input['ref_values'] = ref_images
    
    return val_input


def refining_ref_with_single_pipe(start_idx, ref_images, val_input, weight_dtype):
    keys = ['meta_data', 'captions_clip', 'captions_t5', "pixel_values", "camera_param", 'kwargs', "bev_map_with_aux", "bev_hdmap", "layout_canvas"]
    val_input_single = {}
    for key in keys:
        if key == 'meta_data':
            meta_data_keys = ['metas', 'lidar2image', 'gt_bboxes_3d', 'gt_bboxes_3d']
            val_input_single[key] = {meta_key: val_input[key][meta_key][start_idx:start_idx+2] for meta_key in meta_data_keys}
        elif key == 'kwargs':
            val_input_single[key] = {'bboxes_3d_data': {k:v[start_idx:start_idx+2] for k,v in val_input[key]['bboxes_3d_data'].items()}}
        else:
            val_input_single[key] = val_input[key][start_idx:start_idx+2]
    val_input['conditional_values'] = torch.stack(ref_images)

    cfg_single.runner.validation_times = 1
    cfg_single.show_box = False
    cfg_single.show_lane = False
    gen_imgs_list = run_one_batch_img(cfg_single, pipe_single, val_input_single, weight_dtype)[4]
    ref_image = [torch.stack([ImageNormalize()(x) for x in gen_imgs_list[idx][0]]) for idx in range(2)]

    return ref_image


def load_pipe(resume_from_checkpoint, config_name="test_config_img", device='cuda'):
    try:
        initialize(version_base=None, config_path="../configs")
    except:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="../configs")
    output_dir = to_absolute_path(resume_from_checkpoint)
    original_overrides = OmegaConf.load(
        os.path.join(output_dir, "hydra/overrides.yaml"))
    overrides = original_overrides
    cfg = compose(config_name, overrides=overrides)
    cfg.resume_from_checkpoint = resume_from_checkpoint

    #### model ####
    assert cfg.resume_from_checkpoint is not None, "Please set model to load"
    pipe, weight_dtype = build_pipe(cfg, device)
    update_progress_bar_config(pipe, leave=False)

    return pipe, cfg, weight_dtype


def main():
    os.makedirs(cfg.log_root, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.log_root, "run_config.yaml"))
    #### setup everything ####
    val_param = OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
    val_dataset = build_dataset(val_param)

    #### for each scene
    scene_idx = 0
    for scene, scene_index in val_dataset.scene_key_frame.items():
        scene_idx += 1
        cfg.runner.validation_index = scene_index
        logging.info(f"Your validation index: {cfg.runner.validation_index}")
        val_dataloader = load_data(cfg, pipe)

        #### start ####
        total_num = 0
        batch_index = 0
        progress_bar = tqdm(
            range(len(val_dataloader) * cfg.runner.validation_times),
            desc="Steps",
        )
        os.makedirs(os.path.join(cfg.log_root, scene, "frames"), exist_ok=True)
        all_ori_img_paths = []
        all_gen_img_paths = []
        ref_images = []
        ref_can_bus = []
        # TODO: you can change the overlap_length to adpat to long videos
        # Besieds, it's better to let the overlapped frames use the same init_noise
        overlap_length = args.overlap_length
        generator = None
        for val_input in val_dataloader:
            batch_index += 1
            batch_img_index = 0
            ori_img_paths = []
            gen_img_paths = []
            if batch_index > 1:
                val_input['ref_values'] = torch.stack(ref_images)
                if prev_pos is not None:
                    val_input['can_bus'][0][:3] = prev_pos
                    val_input['can_bus'][0][-1] = prev_angle 
                ref_can_bus[0][:3] = 0
                ref_can_bus[0][-1] = 0
                val_input['ref_can_bus'] = torch.stack(ref_can_bus)
                if args.overlap_condition and len(overlap_images):
                    val_input['overlap_values'] = torch.stack(overlap_images)
            else:
                val_input = generate_ref_with_single_pipe(val_input, weight_dtype)
            
            if generator is not None and cfg.get('overlap_same_noise', False):
                temp_generator_state = [x.get_state() for x in generator[-overlap_length:]]
            else:
                temp_generator_state = None
            return_tuples = run_one_batch_video(cfg, pipe, val_input, weight_dtype,
                                        transparent_bg=transparent_bg, generator=generator,
                                        map_size=target_map_size)
            
            generator, return_tuples = return_tuples[-1], return_tuples[:-1]
            if temp_generator_state is not None:
                generator[:overlap_length] = [x.set_state(y) for x,y in zip(generator[:overlap_length], temp_generator_state)]

            # ref_idxs = sorted(random.sample(range(cfg.dataset.data.val.candidate_length), cfg.model.ref_length))
            ref_idxs = [cfg.model.video_length-overlap_length-2, cfg.model.video_length-overlap_length-1]
            ref_images = [torch.stack([ImageNormalize()(x) for x in return_tuples[4][idx][0]]) for idx in ref_idxs]

            if args.refinement:
                assert ref_idxs[0] == ref_idxs[1] - 1
                ref_images = refining_ref_with_single_pipe(ref_idxs[0], ref_images, val_input, weight_dtype)

            if args.overlap_condition:
                overlap_images = [torch.stack([ImageNormalize()(x) for x in return_tuples[4][idx][0]]) for idx in range(cfg.model.video_length-overlap_length, cfg.model.video_length)]
            ref_can_bus = [val_input['can_bus'][idx] for idx in ref_idxs]

            if overlap_length>0:
                prev_pos, prev_angle = val_input['ref_can_bus'][0][:3], val_input['ref_can_bus'][0][-1]
                for idx in range(ref_idxs[-1]+1, cfg.model.video_length-overlap_length+1):
                    prev_pos += val_input['can_bus'][idx][:3]
                    prev_angle += val_input['can_bus'][idx][-1]
            else:
                prev_pos = None

            for map_img, ori_imgs, ori_imgs_wb, ori_imgs_wl, gen_imgs_list, gen_imgs_wb_list, gen_imgs_wl_list in zip(*return_tuples):
                # save map
                map_img.save(os.path.join(
                    cfg.log_root, scene, "frames",
                    f"{batch_index}_{batch_img_index}_map_{total_num}.png"))

                # save ori
                if ori_imgs is not None:
                    ori_img = output_func(ori_imgs)
                    save_path = os.path.join(
                        cfg.log_root, scene, "frames",
                        f"{batch_index}_{batch_img_index}_ori_{total_num}.png")
                    ori_img.save(save_path)
                    ori_img_paths.append(save_path)

                # save gen
                gen_img = output_func(gen_imgs_list[0])
                save_path = os.path.join(
                    cfg.log_root, scene, "frames",
                    f"{batch_index}_{batch_img_index}_gen_{total_num}.png")
                gen_img.save(save_path)
                gen_img_paths.append(save_path)

                if cfg.show_box:
                    # save ori with box
                    if ori_imgs_wb is not None:
                        ori_img_with_box = output_func(ori_imgs_wb)
                        ori_img_with_box.save(os.path.join(
                            cfg.log_root, scene, "frames",
                            f"{batch_index}_{batch_img_index}_ori_box_{total_num}.png"))
                    # save gen with box
                    gen_img_with_box = output_func(gen_imgs_wb_list[0])
                    gen_img_with_box.save(os.path.join(
                        cfg.log_root, scene, "frames",
                        f"{batch_index}_{batch_img_index}_gen_box_{total_num}.png"))

                if cfg.show_lane:
                    # save ori with lane
                    if ori_imgs_wl is not None:
                        ori_img_with_lane = output_func(ori_imgs_wl)
                        ori_img_with_lane.save(os.path.join(
                            cfg.log_root, scene, "frames",
                            f"{batch_index}_{batch_img_index}_ori_lane_{total_num}.png"))
                    # save gen with lane
                    gen_img_with_lane = output_func(gen_imgs_wl_list[0])
                    gen_img_with_lane.save(os.path.join(
                        cfg.log_root, scene, "frames",
                        f"{batch_index}_{batch_img_index}_gen_lane_{total_num}.png"))

                total_num += 1
                batch_img_index += 1

            # make video
            make_video_with_filenames(
                ori_img_paths, os.path.join(
                    cfg.log_root, scene, f"{batch_index}_{batch_img_index}_ori.mp4"),
                fps=cfg.fps)
            make_video_with_filenames(
                gen_img_paths, os.path.join(
                    cfg.log_root, scene, f"{batch_index}_{batch_img_index}_gen.mp4"),
                fps=cfg.fps)

            if batch_index > 1:
                ori_img_paths = ori_img_paths[overlap_length:]
                gen_img_paths = gen_img_paths[overlap_length:]
            all_ori_img_paths.extend(ori_img_paths)
            all_gen_img_paths.extend(gen_img_paths)

            # update bar
            progress_bar.update(cfg.runner.validation_times)

        prefix = '-'.join([str(x) for x in cfg.runner.validation_index])
        make_video_with_filenames(
            all_ori_img_paths, os.path.join(
                cfg.log_root, scene, f"{prefix}_ori.mp4"),
            fps=cfg.fps)
        make_video_with_filenames(
            all_gen_img_paths, os.path.join(
                cfg.log_root, scene, f"{prefix}_gen{'_overlap_noise' if cfg.get('overlap_same_noise', False) else ''}{'_overlap_condition' if args.overlap_condition else ''}.mp4"), fps=cfg.fps)


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_img", type=str, default='pretrained/x-scene-img_224x400')
    parser.add_argument("--model_video", type=str, default='pretrained/x-scene-video_224x400')
    parser.add_argument("--overlap_condition", action='store_true')
    parser.add_argument("--refinement", action='store_true')
    parser.add_argument("--show_box", action='store_true')
    parser.add_argument("--show_lane", action='store_true')
    parser.add_argument("--output", type=str, default='./work_dirs/test_video')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()
    pipe_single, cfg_single, _ = load_pipe(args.model_img, config_name="test_config_img")
    pipe, cfg, weight_dtype = load_pipe(args.model_video, config_name="test_config_video")
    cfg.show_box = args.show_box
    cfg.show_lane = args.show_lane
    cfg.log_root = os.path.join(args.output, str(time.time()))
    cfg.fps = 12
    args.overlap_length = 1
    main()
