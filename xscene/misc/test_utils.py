from typing import Tuple, Union, List
import os
import logging
from hydra.core.hydra_config import HydraConfig
from functools import partial
from omegaconf import OmegaConf
from omegaconf import DictConfig
from PIL import Image
from scipy import stats
from collections import defaultdict

import numpy as np
import numba
from numba import typed
import torch
from torchvision.transforms.functional import to_pil_image
from einops import rearrange

from mmdet3d.datasets import build_dataset
from diffusers import UniPCMultistepScheduler, AutoencoderKL
import accelerate
from accelerate.utils import set_seed

from xscene.dataset import collate_fn, ListSetWrapper, FolderSetWrapper
from xscene.pipeline.pipeline_bev_controlnet import (
    StableDiffusionBEVControlNetPipeline,
    BEVStableDiffusionPipelineOutput,
)
from xscene.pipeline.pipeline_occ_unet import StableDiffusionOccUNetPipeline
from xscene.runner.utils import (
    visualize_map, img_m11_to_01, show_box_on_views
)
from xscene.misc.common import load_module
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from xscene.occ_vae.dataset.nuscenes_occ_dataset import nb_process_label
from xscene.occ_vae.networks.networks import AutoEncoderGroupSkip
from xscene.modules.lora import monkeypatch_or_replace_lora_extended
from xscene.networks.utils import make_query
from tools.vis_occ import visualize_occ, draw_occ_triplane, draw_latent_field2D, visualize_planemap


def insert_pipeline_item(cfg: DictConfig, search_type, item=None) -> None:
    if item is None:
        return
    assert OmegaConf.is_list(cfg)
    ori_cfg: List = OmegaConf.to_container(cfg)
    for index, _it in enumerate(cfg):
        if _it['type'] == search_type:
            break
    else:
        raise RuntimeError(f"cannot find type: {search_type}")
    ori_cfg.insert(index + 1, item)
    cfg.clear()
    cfg.merge_with(ori_cfg)


def draw_box_on_imgs(cfg, idx, val_input, ori_imgs, transparent_bg=False) -> Tuple[Image.Image, ...]:
    if transparent_bg:
        in_imgs = [Image.new('RGB', img.size) for img in ori_imgs]
    else:
        in_imgs = ori_imgs
    out_imgs = show_box_on_views(
        OmegaConf.to_container(cfg.dataset.object_classes, resolve=True),
        in_imgs,
        val_input['meta_data']['gt_bboxes_3d'][idx].data,
        val_input['meta_data']['gt_labels_3d'][idx].data.numpy(),
        val_input['meta_data']['lidar2image'][idx].data.numpy(),
        val_input['meta_data']['img_aug_matrix'][idx].data.numpy(),
    )
    if transparent_bg:
        for i in range(len(out_imgs)):
            out_imgs[i].putalpha(Image.fromarray(
                (np.any(np.asarray(out_imgs[i]) > 0, axis=2) * 255).astype(np.uint8)))
    return out_imgs


def draw_lane_on_imgs(image_list, canvas_array):
    if isinstance(canvas_array, torch.Tensor):
        canvas_array = canvas_array.cpu().numpy()
    # image: 6,3,H,W
    # canvas: 6,C,H,W
    image_array = np.array([np.array(image) / 255.0 for image in image_list])

    images_ = []
    colors =[(1,0,0),(0,1,0),(0,0,1)]
    for image, canvas in zip(image_array, canvas_array):
        for category in range(3):
            image[canvas[category] == 1] = [colors[category][0], colors[category][1], colors[category][2]]

        image = (image * 255).astype(np.uint8) 
        images_.append(Image.fromarray(image))

    return images_


def update_progress_bar_config(pipe, **kwargs):
    if hasattr(pipe, "_progress_bar_config"):
        config = pipe._progress_bar_config
        config.update(kwargs)
    else:
        config = kwargs
    pipe.set_progress_bar_config(**config)


def setup_logger_seed(cfg):
    #### setup logger ####
    # only log debug info to log file
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
    # handle log from some packages
    logging.getLogger("shapely.geos").setLevel(logging.WARN)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("accelerate.tracking").setLevel(logging.INFO)
    logging.getLogger("numba").setLevel(logging.WARN)
    logging.getLogger("PIL").setLevel(logging.WARN)
    logging.getLogger("matplotlib").setLevel(logging.WARN)
    setattr(cfg, "log_root", HydraConfig.get().runtime.output_dir)
    set_seed(cfg.seed)


def build_pipe(cfg, device):
    if hasattr(cfg.model, "model_module"):  # img diffusion use float16
        weight_dtype = torch.float16
    else:                                   # occ diffusion use float32
        weight_dtype = torch.float32
    if cfg.resume_from_checkpoint.endswith("/"):
        cfg.resume_from_checkpoint = cfg.resume_from_checkpoint[:-1]
    pipe_param = {}

    # controlnet
    if hasattr(cfg.model, "model_module"):
        model_cls = load_module(cfg.model.model_module)
        controlnet_path = os.path.join(
            cfg.resume_from_checkpoint, cfg.model.controlnet_dir)
        if not os.path.exists(controlnet_path):
            controlnet_path = os.path.join(
                cfg.model.pretrained_img_unet, cfg.model.controlnet_dir)
        logging.info(f"Loading controlnet from {controlnet_path} with {model_cls}")
        controlnet = model_cls.from_pretrained(
            controlnet_path, torch_dtype=weight_dtype).to(device)
        controlnet.eval()  # from_pretrained will set to eval mode by default
        pipe_param["controlnet"] = controlnet

    # unet
    if hasattr(cfg.model, "unet_module"):
        unet_cls = load_module(cfg.model.unet_module)
        unet_path = os.path.join(cfg.resume_from_checkpoint, cfg.model.unet_dir)
        logging.info(f"Loading unet from {unet_path} with {unet_cls}")
        unet = unet_cls.from_pretrained(
            unet_path, torch_dtype=weight_dtype).to(device)
        if hasattr(cfg.model, 'sc_attn_index'):
            logging.warn(f"We reset sc_attn_index from config.")
            for mod in unet.modules():
                if hasattr(mod, "_sc_attn_index"):
                    mod._sc_attn_index = OmegaConf.to_container(
                        cfg.model.sc_attn_index, resolve=True)
        unet.eval()
        pipe_param["unet"] = unet

    # scene_embedder_cls
    if hasattr(cfg.model, "scene_embedder_cls"):
        scene_embedder_cls = load_module(cfg.model.scene_embedder_cls)
        scene_embedder_param = OmegaConf.to_container(cfg.model.scene_embedder, resolve=True)
        scene_embedder = scene_embedder_cls(**scene_embedder_param)
        scene_embedder_path = os.path.join(cfg.resume_from_checkpoint, cfg.model.scene_embedder_dir)
        if not os.path.exists(scene_embedder_path):
            scene_embedder_path = os.path.join(
                cfg.model.pretrained_img_unet, cfg.model.scene_embedder_dir)
        logging.info(f"Loading scene_embedder from {scene_embedder_path} with {scene_embedder_cls}")
        state_dict = torch.load(os.path.join(scene_embedder_path, "scene_embedder_model.bin"), map_location='cpu')
        scene_embedder.load_state_dict(state_dict)
        scene_embedder.type(torch.cuda.FloatTensor)
        scene_embedder.eval()
        pipe_param["scene_embedder"] = scene_embedder

    # can_bus_embedder
    if hasattr(cfg.model, "can_bus_embedder"):
        can_bus_embedder_cls = load_module(cfg.model.can_bus_embedder_cls)
        can_bus_embedder_param = OmegaConf.to_container(cfg.model.can_bus_embedder, resolve=True)
        can_bus_embedder = can_bus_embedder_cls(**can_bus_embedder_param)
        can_bus_embedder_path = os.path.join(cfg.resume_from_checkpoint, cfg.model.can_bus_embedder_dir)
        state_dict = torch.load(os.path.join(can_bus_embedder_path, "can_bus_embedder_model.bin"), map_location='cpu')
        can_bus_embedder.load_state_dict(state_dict)
        can_bus_embedder.type(torch.cuda.HalfTensor)
        can_bus_embedder.eval()
        pipe_param["can_bus_embedder"] = can_bus_embedder
        logging.info(f"Loading can_bus_embedder from {can_bus_embedder_path} with {can_bus_embedder_cls}")


    # occ vae
    if hasattr(cfg.model, "vae_ckpt"):
        occ_vae = AutoEncoderGroupSkip.from_pretrained(cfg.model.vae_ckpt)
        occ_vae.to(device=device, dtype=weight_dtype)
        occ_vae.eval()
    # image vae
    else:
        vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae")
        vae.to(device=device, dtype=weight_dtype)
        vae.eval()

    # CLIP
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.to(device=device, dtype=weight_dtype)
    text_encoder.eval()
    
    # T5
    tokenizer_t5 = T5Tokenizer.from_pretrained(cfg.model.pretrained_t5_path, model_max_length=512, ignore_mismatched_sizes=True)
    text_encoder_t5 = T5EncoderModel.from_pretrained(cfg.model.pretrained_t5_path)
    text_encoder_t5.to(device=device, dtype=weight_dtype)
    text_encoder_t5.eval()


    # prepare text_tokenizer and text_encoder for bbox_embedder
    if hasattr(cfg.model, "model_module"):  # multi-view
        pipe_param["controlnet"].prepare(
            cfg, tokenizer=tokenizer, text_encoder=text_encoder)
    else:
        if hasattr(pipe_param["unet"], "bbox_embedder"):
            pipe_param["unet"].prepare(         # occ
                cfg, tokenizer=tokenizer, text_encoder=text_encoder)

    if 'multiview' in cfg.model.unet_module:
        pipe_param.update({
            "vae": vae,
            "text_encoder": text_encoder,
            "text_encoder_t5": text_encoder_t5,
            "tokenizer": tokenizer,
            "tokenizer_t5": tokenizer_t5,
        })
    elif 'occ' in cfg.model.unet_module:
        pipe_param.update({
            "vae": occ_vae,
            "text_encoder": text_encoder_t5 if cfg.model.use_cross_attn_cond else None,
            "tokenizer": tokenizer_t5 if cfg.model.use_cross_attn_cond else None,
            "use_cross_attn_cond": cfg.model.use_cross_attn_cond,
            "use_map_cond": cfg.model.use_map_cond,
        })
    else:
        raise NotImplemented(
            f"Unet module {cfg.model.unet_module} not supported, please check your config file."
        )

    pipe_cls = load_module(cfg.model.pipe_module)
    logging.info(f"Build pipeline with {pipe_cls}")
    pipe = pipe_cls.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        **pipe_param,
        safety_checker=None,
        feature_extractor=None,  # since v1.5 has default, we need to override
        torch_dtype=weight_dtype
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    if cfg.runner.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    pipe = pipe.to(device)

    # when inference, memory is not the issue. we do not need this.
    # pipe.enable_model_cpu_offload()
    return pipe, weight_dtype


def prepare_all(cfg, device='cuda', need_loader=True):
    assert cfg.resume_from_checkpoint is not None, "Please set model to load"
    setup_logger_seed(cfg)

    #### model ####
    pipe, weight_dtype = build_pipe(cfg, device)
    update_progress_bar_config(pipe, leave=False)

    if not need_loader:
        return pipe, weight_dtype

    #### datasets ####
    if cfg.runner.validation_index == "demo":
        val_dataset = FolderSetWrapper("demo/data")
    else:
        val_dataset = build_dataset(
            OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
        )
        if cfg.runner.validation_index != "all":
            val_dataset = ListSetWrapper(
                val_dataset, cfg.runner.validation_index)

    #### dataloader ####
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
        "map_cond_type": cfg.model.get('map_cond_type', 'bev_seg'),
    }
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=partial(collate_fn, is_train=False, **collate_fn_param),
        batch_size=cfg.runner.validation_batch_size,
        num_workers=cfg.runner.num_workers,
    )
    return pipe, val_dataloader, weight_dtype


def new_local_seed(global_generator):
    local_seed = torch.randint(
        0x7ffffffffffffff0, [1], generator=global_generator).item()
    logging.debug(f"Using seed: {local_seed}")
    return local_seed


def run_one_batch_pipe_img(
    cfg,
    pipe: StableDiffusionBEVControlNetPipeline,
    pixel_values: torch.FloatTensor,  # useless
    captions: Union[str, List[str]],
    captions_t5: Union[str, List[str]],
    bev_map_with_aux: torch.FloatTensor,
    camera_param: Union[torch.Tensor, None],
    bev_controlnet_kwargs: dict,
    meta_data,
    global_generator=None
):
    """call pipe several times to generate images

    Args:
        cfg (_type_): _description_
        pipe (StableDiffusionBEVControlNetPipeline): _description_
        captions (Union[str, List[str]]): _description_
        bev_map_with_aux (torch.FloatTensor): (B=1, C=26, 200, 200), float32
        camera_param (Union[torch.Tensor, None]): (B=1, N=6, 3, 7), if None, 
            use learned embedding for uncond_cam

    Returns:
        List[List[List[Image.Image]]]: 3-dim list of PIL Image: B, Times, views
    """
    # for each input param, we generate several times to check variance.
    if isinstance(captions, str):
        batch_size = 1
    else:
        batch_size = len(captions)

    # let different prompts have the same random seed
    if cfg.seed is None:
        generator = None
    else:
        if global_generator is not None:
            if cfg.fix_seed_within_batch:
                generator = []
                for _ in range(batch_size):
                    local_seed = new_local_seed(global_generator)
                    generator.append(torch.manual_seed(local_seed))
            else:
                local_seed = new_local_seed(global_generator)
                generator = torch.manual_seed(local_seed)
        else:
            if cfg.fix_seed_within_batch:
                generator = [torch.manual_seed(cfg.seed)
                             for _ in range(batch_size)]
            else:
                generator = torch.manual_seed(cfg.seed)

    gen_imgs_list = [[] for _ in range(batch_size)]
    for ti in range(cfg.runner.validation_times):
        image: BEVStableDiffusionPipelineOutput = pipe(
            prompt=captions,
            prompt_t5=captions_t5,
            image=bev_map_with_aux,
            camera_param=camera_param,
            height=cfg.dataset.image_size[0],
            width=cfg.dataset.image_size[1],
            generator=generator,
            bev_controlnet_kwargs=bev_controlnet_kwargs,
            img_metas=meta_data,
            **cfg.runner.pipeline_param,
        )
        image: List[List[Image.Image]] = image.images
        for bi, imgs in enumerate(image):
            gen_imgs_list[bi].append(imgs)
    return gen_imgs_list

def run_one_batch_img(cfg, pipe, val_input, weight_dtype, global_generator=None,
                  run_one_batch_pipe_func=run_one_batch_pipe_img,
                  transparent_bg=False, map_size=400):
    """Run one batch of data according to your configuration

    Returns:
        List[Image.Image]: map image
        List[List[Image.Image]]: ori images
        List[List[Image.Image]]: ori images with bbox, can be []
        List[List[Tuple[Image.Image]]]: generated images list
        List[List[Tuple[Image.Image]]]: generated images list, can be []
        if 2-dim: B, views; if 3-dim: B, Times, views
    """
    bs = len(val_input['meta_data']['metas'])

    # TODO: not sure what to do with filenames
    # image_names = [val_input['meta_data']['metas'][i].data['filename']
    #                for i in range(bs)]
    logging.debug(f"Caption: {val_input['captions_clip']}")

    # map
    map_imgs = []
    if "bev_map_with_aux" in val_input:
        for bev_map in val_input["bev_map_with_aux"]:
            map_img_np = visualize_map(cfg, bev_map, target_size=map_size)
            map_imgs.append(Image.fromarray(map_img_np))

    # ori
    ori_imgs = [None for bi in range(bs)]
    ori_imgs_with_box = [None for bi in range(bs)]
    ori_imgs_with_lane = [None for bi in range(bs)]
    if val_input["pixel_values"] is not None:
        ori_imgs = [
            [to_pil_image(img_m11_to_01(val_input["pixel_values"][bi][i]))
             for i in range(6)] for bi in range(bs)
        ]
        if cfg.show_box:
            ori_imgs_with_box = [
                draw_box_on_imgs(cfg, bi, val_input, ori_imgs[bi],
                                 transparent_bg=transparent_bg)
                for bi in range(bs)
            ]
        if cfg.show_lane:
            ori_imgs_with_lane = [
                draw_lane_on_imgs(ori_imgs[bi], val_input["layout_canvas"][bi])
                for bi in range(bs)
            ]

    # camera_emb = self._embed_camera(val_input["camera_param"])
    camera_param = val_input["camera_param"].to(weight_dtype)

    controlnet_image = val_input["bev_map_with_aux"].to(
        dtype=weight_dtype)
    controlnet_image = [controlnet_image]
    # layout canvas
    if 'with_layout_canvas' in cfg.model.controlnet and cfg.model.controlnet.with_layout_canvas:
        bev_hdmap = val_input["bev_hdmap"].to(dtype=weight_dtype)
        layout_canvas = val_input["layout_canvas"].to(dtype=weight_dtype)
        controlnet_image += [bev_hdmap, layout_canvas]
    if 'with_occ_render_img' in cfg.model.controlnet and cfg.model.controlnet.with_occ_render_img:
        occ_render_img = val_input["occ_render_image"].to(dtype=weight_dtype)
        occ_render_depth = val_input["occ_render_depth"].to(dtype=weight_dtype)
        controlnet_image += [occ_render_img, occ_render_depth]

    # 3-dim list: B, Times, views
    gen_imgs_list = run_one_batch_pipe_func(
        cfg, pipe, val_input['pixel_values'], val_input['captions_clip'], val_input['captions_t5'],
        controlnet_image, camera_param, val_input['kwargs'], val_input['meta_data'],
        global_generator=global_generator)

    # save gen with box
    gen_imgs_wb_list = [None for _ in gen_imgs_list]
    if cfg.show_box:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wb_list[bi] = [
                draw_box_on_imgs(cfg, bi, val_input, images[ti],
                                 transparent_bg=transparent_bg)
                for ti in range(len(images))
            ]
    # save gen with lane
    gen_imgs_wl_list = [None for _ in gen_imgs_list]
    if cfg.show_lane:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wl_list[bi] = [
                draw_lane_on_imgs(gen_imgs_wb_list[bi][ti], val_input["layout_canvas"][bi])
                for ti in range(len(images))
            ]

    return map_imgs, ori_imgs, ori_imgs_with_box, ori_imgs_with_lane, gen_imgs_list, gen_imgs_wb_list, gen_imgs_wl_list

def run_one_batch_pipe_video(
    cfg,
    pipe: StableDiffusionBEVControlNetPipeline,
    ref_values: torch.FloatTensor,
    captions: Union[str, List[str]],
    captions_t5: Union[str, List[str]],
    bev_map_with_aux: torch.FloatTensor,
    can_bus: torch.FloatTensor,
    camera_param: Union[torch.Tensor, None],
    bev_controlnet_kwargs: dict,
    meta_data=None,
    overlap_values=None,
    global_generator=None,
    generator=None,
):
    """call pipe several times to generate images

    Args:
        cfg (_type_): _description_
        pipe (StableDiffusionBEVControlNetPipeline): _description_
        captions (Union[str, List[str]]): _description_
        bev_map_with_aux (torch.FloatTensor): (B=1, C=26, 200, 200), float32
        camera_param (Union[torch.Tensor, None]): (B=1, N=6, 3, 7), if None, 
            use learned embedding for uncond_cam

    Returns:
        List[List[List[Image.Image]]]: 3-dim list of PIL Image: B, Times, views
    """
    # for each input param, we generate several times to check variance.
    if isinstance(captions, str):
        batch_size = 1
    else:
        batch_size = len(captions)

    # let different prompts have the same random seed
    device = pipe.device
    if generator is None and cfg.seed is not None:
        if global_generator is not None:
            if cfg.fix_seed_within_batch:
                generator = []
                for _ in range(batch_size):
                    local_seed = new_local_seed(global_generator)
                    generator.append(torch.Generator(
                        device=device).manual_seed(local_seed))
            else:
                local_seed = new_local_seed(global_generator)
                generator = torch.Generator(
                    device=device).manual_seed(local_seed)
        else:
            if cfg.fix_seed_within_batch:
                generator = [
                    torch.Generator(device=device).manual_seed(cfg.seed)
                    for _ in range(batch_size)]
            elif 'overlap_same_noise' in cfg and cfg.overlap_same_noise:
                generator = [
                    torch.Generator(device=device).manual_seed(cfg.seed+i)
                    for i in range(batch_size)]
            else:
                generator = torch.Generator(device=device).manual_seed(cfg.seed)

    if overlap_values is not None:
        overlap_length, N_cam = overlap_values.shape[:2]
        with torch.no_grad():
            latents = pipe.vae.encode(
                rearrange(overlap_values, "b n c h w -> (b n) c h w").to(
                    dtype=pipe.vae.dtype, device=pipe._execution_device
                )
            ).latent_dist.mean
            latents = latents * pipe.vae.config.scaling_factor
            latents = rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam)

    pipeline_param = {k: v for k, v in cfg.runner.pipeline_param.items()}
    gen_imgs_list = [[] for _ in range(batch_size)]
    for ti in range(cfg.runner.validation_times):
        if overlap_values is not None:
            conditional_latents = [[None,] * N_cam for _ in range(batch_size)]
            for b in range(overlap_length):
                for j in range(N_cam):
                    conditional_latents[b][j] = latents[b, j]
        else:
            conditional_latents = None

        if cfg.runner.pipeline_param.init_noise == "both":
            if pipeline_param['init_noise'] == "both":
                pipeline_param['init_noise'] = "same"
            elif pipeline_param['init_noise'] == "same":
                pipeline_param['init_noise'] = "rand"
            elif pipeline_param['init_noise'] == "rand":
                pipeline_param['init_noise'] = "same"

        image: BEVStableDiffusionPipelineOutput = pipe(
            prompt=captions,
            prompt_t5=captions_t5,
            image=bev_map_with_aux,
            ref_image=ref_values,
            can_bus=can_bus,
            camera_param=camera_param,
            height=cfg.dataset.image_size[0],
            width=cfg.dataset.image_size[1],
            conditional_latents=conditional_latents,
            generator=generator,
            bev_controlnet_kwargs=bev_controlnet_kwargs,
            img_metas=meta_data,
            **pipeline_param,
        )
        image: List[List[Image.Image]] = image.images
        for bi, imgs in enumerate(image):
            gen_imgs_list[bi].append(imgs)
    return gen_imgs_list, generator

def run_one_batch_video(cfg, pipe, val_input, weight_dtype, generator=None,
                  run_one_batch_pipe_func=run_one_batch_pipe_video,
                  transparent_bg=False, map_size=400):
    """Run one batch of data according to your configuration

    Returns:
        List[Image.Image]: map image
        List[List[Image.Image]]: ori images
        List[List[Image.Image]]: ori images with bbox, can be []
        List[List[Tuple[Image.Image]]]: generated images list
        List[List[Tuple[Image.Image]]]: generated images list, can be []
        if 2-dim: B, views; if 3-dim: B, Times, views
    """
    bs = len(val_input['meta_data']['metas'])

    # TODO: not sure what to do with filenames
    # image_names = [val_input['meta_data']['metas'][i].data['filename']
    #                for i in range(bs)]
    logging.debug(f"Caption: {val_input['captions_clip']}")

    # map
    map_imgs = []
    if "bev_map_with_aux" in val_input:
        for bev_map in val_input["bev_map_with_aux"]:
            map_img_np = visualize_map(cfg, bev_map, target_size=map_size)
            map_imgs.append(Image.fromarray(map_img_np))

    # ori
    ori_imgs = [None for bi in range(bs)]
    ori_imgs_with_box = [None for bi in range(bs)]
    ori_imgs_with_lane = [None for bi in range(bs)]
    if val_input["pixel_values"] is not None:
        ori_imgs = [
            [to_pil_image(img_m11_to_01(val_input["pixel_values"][bi][i]))
             for i in range(6)] for bi in range(bs)
        ]
        if cfg.show_box:
            ori_imgs_with_box = [
                draw_box_on_imgs(cfg, bi, val_input, ori_imgs[bi],
                                 transparent_bg=transparent_bg)
                for bi in range(bs)
            ]
        if cfg.show_lane:
            ori_imgs_with_lane = [
                draw_lane_on_imgs(ori_imgs[bi], val_input["layout_canvas"][bi])
                for bi in range(bs)
            ]

    # camera_emb = self._embed_camera(val_input["camera_param"])
    camera_param = val_input["camera_param"].to(weight_dtype)
    ref_image = val_input["ref_values"].to(weight_dtype)
    ref_can_bus = val_input["ref_can_bus"].to(weight_dtype)
    can_bus =  val_input["can_bus"].to(weight_dtype)
    can_bus = torch.cat([ref_can_bus, can_bus], dim=0)
    overlap_values = val_input['overlap_values'].to(weight_dtype) if 'overlap_values' in val_input else None

    controlnet_image = val_input["bev_map_with_aux"].to(
        dtype=weight_dtype)
    controlnet_image = [controlnet_image]
    # layout canvas
    if 'with_layout_canvas' in cfg.model.controlnet and cfg.model.controlnet.with_layout_canvas:
        bev_hdmap = val_input["bev_hdmap"].to(dtype=weight_dtype)
        layout_canvas = val_input["layout_canvas"].to(dtype=weight_dtype)
        controlnet_image += [bev_hdmap, layout_canvas]
    if 'with_occ_render_img' in cfg.model.controlnet and cfg.model.controlnet.with_occ_render_img:
        occ_render_img = val_input["occ_render_image"].to(dtype=weight_dtype)
        occ_render_depth = val_input["occ_render_depth"].to(dtype=weight_dtype)
        controlnet_image += [occ_render_img, occ_render_depth]

    # 3-dim list: B, Times, views
    gen_imgs_list, generator = run_one_batch_pipe_func(
        cfg, pipe, ref_image, val_input['captions_clip'], val_input['captions_t5'],
        controlnet_image, can_bus, camera_param, val_input['kwargs'], val_input['meta_data'],
        overlap_values=overlap_values, generator=generator)

    # save gen with box
    gen_imgs_wb_list = [None for _ in gen_imgs_list]
    if cfg.show_box:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wb_list[bi] = [
                draw_box_on_imgs(cfg, bi, val_input, images[ti],
                                 transparent_bg=transparent_bg)
                for ti in range(len(images))
            ]
    # save gen with lane
    gen_imgs_wl_list = [None for _ in gen_imgs_list]
    if cfg.show_lane:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wl_list[bi] = [
                draw_lane_on_imgs(gen_imgs_wb_list[bi][ti], val_input["layout_canvas"][bi])
                for ti in range(len(images))
            ]

    return map_imgs, ori_imgs, ori_imgs_with_box, ori_imgs_with_lane, gen_imgs_list, gen_imgs_wb_list, gen_imgs_wl_list, generator

@numba.njit
def fast_mode_numba(data):
    n_rows, n_cols = data.shape
    modes = np.empty(n_cols, dtype=data.dtype)

    for i in range(n_cols):
        counter = typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.int64,
        )
        for j in range(n_rows):
            val = data[j, i]
            if val in counter:
                counter[val] += 1
            else:
                counter[val] = 1
        
        max_count = -1
        mode = -1
        for key in counter:
            if counter[key] > max_count:
                max_count = counter[key]
                mode = key

        modes[i] = mode

    return modes

def process_voxel_list(voxel_list):
    """
    Process a list of voxel arrays by:
    1. Finding the same xyz coordinates across different elements
    2. Computing the mode (most frequent class) for each unique xyz coordinate
    3. Filtering out coordinates that appear less than N//2 times
    
    Parameters:
    -----------
    voxel_list: list of numpy.ndarray
        List of length N, where each element is an array of shape (M_i, 4)
        Each row in the array contains [x, y, z, class_label]
        Note: M_i may vary between elements
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (M_out, 4) containing [x, y, z, mode_class] for filtered voxels
    """
    N = len(voxel_list)
    threshold = N // 2
    
    # Dictionary to store class labels for each unique xyz coordinate
    coord_to_classes = defaultdict(list)
    
    # Collect all class labels for each unique xyz coordinate
    for voxel_array in voxel_list:
        for x, y, z, class_label in voxel_array:
            # Use tuple as dictionary key since it's hashable
            coord = (float(x), float(y), float(z))
            coord_to_classes[coord].append(int(class_label))
    
    # Process each coordinate: find mode and filter by frequency
    result = []
    for coord, classes in coord_to_classes.items():
        if len(classes) > threshold:  # Only keep coordinates appearing more than N//2 times
            # Calculate mode (most common class label)
            class_counts = {}
            for c in classes:
                class_counts[c] = class_counts.get(c, 0) + 1
            
            mode_class = max(class_counts.items(), key=lambda x: x[1])[0]
            
            # Add to result: [x, y, z, mode_class]
            result.append([coord[0], coord[1], coord[2], mode_class])
    
    return np.array(result) if result else np.zeros((0, 4))


def run_one_batch_pipe_occ(
    cfg,
    pipe: StableDiffusionOccUNetPipeline,
    captions_t5: Union[str, List[str]],
    triplane_map: torch.FloatTensor,
    query: torch.FloatTensor,
    coords: torch.FloatTensor,
    bev_controlnet_kwargs: dict,
    global_generator=None,
    vis=False,
):
    """call pipe several times to generate images

    Args:
        cfg (_type_): _description_
        pipe (StableDiffusionBEVControlNetPipeline): _description_
        captions (Union[str, List[str]]): _description_
        bev_map_with_aux (torch.FloatTensor): (B=1, C=26, 200, 200), float32
        camera_param (Union[torch.Tensor, None]): (B=1, N=6, 3, 7), if None, 
            use learned embedding for uncond_cam

    Returns:
        List[List[List[Image.Image]]]: 3-dim list of PIL Image: B, Times, views
    """
    # for each input param, we generate several times to check variance.
    if isinstance(captions_t5, str):
        batch_size = 1
    else:
        batch_size = len(captions_t5)

    # let different prompts have the same random seed
    if cfg.seed is None:
        generator = None
    else:
        if global_generator is not None:
            if cfg.fix_seed_within_batch:
                generator = []
                for _ in range(batch_size):
                    local_seed = new_local_seed(global_generator)
                    generator.append(torch.manual_seed(local_seed))
            else:
                local_seed = new_local_seed(global_generator)
                generator = torch.manual_seed(local_seed)
        else:
            if cfg.fix_seed_within_batch:
                generator = [torch.manual_seed(cfg.seed)
                             for _ in range(batch_size)]
            else:
                generator = torch.manual_seed(cfg.seed)

    gen_occ_list = [[] for _ in range(batch_size)]
    gen_coord_occ_list = [[] for _ in range(batch_size)]
    vis_gen_occ_list = [[] for _ in range(batch_size)]
    vis_gen_latent_list = [[] for _ in range(batch_size)]
    for ti in range(cfg.runner.validation_times):
        latents, input_latents, occupancy, voxcoord_with_occupancy = pipe(
            prompt=captions_t5,
            image=triplane_map,
            height=cfg.dataset.tri_size[0]+cfg.dataset.tri_size[2],   # X+Z 100+16
            width=cfg.dataset.tri_size[1]+cfg.dataset.tri_size[2],    # Y+Z 100+16
            query=query.to(pipe.device),
            coords=coords.to(pipe.device),
            generator=generator,
            bev_controlnet_kwargs=bev_controlnet_kwargs,
            **cfg.runner.pipeline_param,
        )

        for bi, (occ, coord_occ, latent) in enumerate(zip(occupancy, voxcoord_with_occupancy, latents)):
            gen_occ_list[bi].append(occ)
            if vis:
                mask = coord_occ[..., -1] != 0
                gen_coord_occ_list[bi].append(coord_occ[mask])
                occupancy_img = draw_occ_triplane(occ, show=False)
                latent_img = draw_latent_field2D(latent, show=False, title="Output")
                vis_gen_occ_list[bi].append(to_pil_image(occupancy_img))
                vis_gen_latent_list[bi].append(to_pil_image(latent_img))

    return gen_occ_list, gen_coord_occ_list, vis_gen_occ_list, vis_gen_latent_list

def run_one_batch_occ(cfg, pipe, val_input, weight_dtype, global_generator=None,
                  run_one_batch_pipe_func=run_one_batch_pipe_occ, map_size=400, vis=False):
    """Run one batch of data according to your configuration

    Returns:
        List[Image.Image]: map image
        List[List[Image.Image]]: ori images
        List[List[Image.Image]]: ori images with bbox, can be []
        List[List[Tuple[Image.Image]]]: generated images list
        List[List[Tuple[Image.Image]]]: generated images list, can be []
        if 2-dim: B, views; if 3-dim: B, Times, views
    """
    bs = len(val_input['meta_data']['metas'])

    # TODO: not sure what to do with filenames
    # image_names = [val_input['meta_data']['metas'][i].data['filename']
    #                for i in range(bs)]
    logging.debug(f"Caption: {val_input['captions_clip']}")

    # map
    map_imgs = []
    if "bev_map_with_aux" in val_input and vis:
        for bev_map in val_input["bev_map_with_aux"]:
            map_img_np = visualize_map(cfg, bev_map, target_size=map_size)
            map_imgs.append(Image.fromarray(map_img_np))
    
    # grid_size
    grid_size = np.array([
        (cfg.dataset.map_bound.x[1] - cfg.dataset.map_bound.x[0]) / cfg.dataset.map_bound.x[2],
        (cfg.dataset.map_bound.y[1] - cfg.dataset.map_bound.y[0]) / cfg.dataset.map_bound.y[2],
        (cfg.dataset.map_bound.z[1] - cfg.dataset.map_bound.z[0]) / cfg.dataset.map_bound.z[2],
    ]).astype(np.int16)
    pipe.grid_size = grid_size

    # construc query
    coords, query = make_query(grid_size)
    batch_size = len(val_input['triplane'])
    coords = coords.repeat(batch_size, 1, 1)
    query = query.repeat(batch_size, 1, 1)

    # 3-dim list: B, Times, views
    gen_occ_list, gen_coord_occ_list, vis_gen_occ_list, vis_gen_latent_list = run_one_batch_pipe_func(
        cfg, pipe, val_input['captions_t5'], val_input['triplane_map'],
        query, coords, val_input['kwargs'], global_generator=global_generator, vis=vis)

    # TTA: output the mode_label for each batch at validation_times dim
    output_gen_occ_list = []    # bs, [h,w,d]
    for gen_occ in (gen_occ_list):  # bs
        gen_occ = np.stack(gen_occ, axis=0) # val_time,h,w,d
        N, h, w, d = gen_occ.shape
        gen_occ = gen_occ.reshape(N, -1)
        # modes, _ = stats.mode(gen_occ, axis=0, keepdims=False)
        modes = fast_mode_numba(gen_occ)
        result = modes.reshape(h, w, d)
        output_gen_occ_list.append([result])
    # TTA: filter floaters for visualization
    output_gen_coord_occ_list = []    # bs, [N,4]
    if vis:
        for gen_coord_occ in (gen_coord_occ_list):  # bs
            result = process_voxel_list(gen_coord_occ)
            output_gen_coord_occ_list.append([result])
    
    # save original input
    target_occ_list = [[] for _ in vis_gen_occ_list]
    vis_target_occ_list = [[] for _ in vis_gen_occ_list]
    vis_target_latent_list = [[] for _ in vis_gen_latent_list]
    for bi, (occ, latent) in enumerate(zip(val_input["triplane_map"], val_input["triplane"])):
        target_occ_list[bi].append(occ)
        if vis:
            target_occ_img = visualize_planemap(occ, show=False)
            target_latent_img = draw_latent_field2D(latent, show=False, title="Target")
            vis_target_occ_list[bi].append(to_pil_image(target_occ_img))
            vis_target_latent_list[bi].append(to_pil_image(target_latent_img))

    return output_gen_occ_list, output_gen_coord_occ_list, target_occ_list, \
        map_imgs, vis_target_occ_list, vis_target_latent_list, vis_gen_occ_list, vis_gen_latent_list
