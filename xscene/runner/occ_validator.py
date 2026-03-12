from typing import Tuple, List
import logging
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image, to_tensor

from diffusers import UniPCMultistepScheduler
from accelerate.tracking import GeneralTracker

from xscene.runner.utils import (
    visualize_map,
    img_m11_to_01,
    concat_6_views,
)
from xscene.misc.common import move_to
from xscene.misc.test_utils import draw_box_on_imgs
from xscene.pipeline.pipeline_bev_controlnet import (
    BEVStableDiffusionPipelineOutput,
)
from xscene.dataset.utils import collate_fn
from xscene.networks.utils import make_query
from tools.vis_occ import visualize_occ, draw_occ_triplane, draw_latent_field2D, visualize_planemap


def format_ori_with_gen(ori_img, gen_img_list, input_img_list=None, nrow=1, to_array=True):
    formatted_images = []

    # first image is input, followed by generations.
    formatted_images.append(np.asarray(ori_img))

    if input_img_list is not None:
        for image in input_img_list:
            formatted_images.append(np.asarray(image))

    for image in gen_img_list:
        formatted_images.append(np.asarray(image))

    # formatted_images = np.stack(formatted_images)
    # 0-255 np -> 0-1 tensor -> grid -> 0-255 pil -> np
    formatted_images = torchvision.utils.make_grid(
        [to_tensor(im) for im in formatted_images], nrow=nrow)
    formatted_images = np.asarray(
        to_pil_image(formatted_images)) if to_array else to_pil_image(formatted_images)
    return formatted_images


class OccValidator:
    def __init__(self, cfg, val_dataset, pipe_cls, pipe_param) -> None:
        self.cfg = cfg
        self.val_dataset = val_dataset
        self.pipe_cls = pipe_cls
        self.pipe_param = pipe_param
        logging.info(
            f"[OccValidator] Validator use model_param: {pipe_param.keys()}")

    def validate(
        self,
        unet,
        trackers: Tuple[GeneralTracker, ...],
        step, weight_dtype, device
    ):
        logging.info("[OccValidator] Running validation... ")
        unet.eval()
        if self.cfg.model.use_cross_attn_cond:
            self.pipe_param["text_encoder"].eval()

        grid_size = np.array([
            (self.cfg.dataset.map_bound.x[1] - self.cfg.dataset.map_bound.x[0]) / self.cfg.dataset.map_bound.x[2],
            (self.cfg.dataset.map_bound.y[1] - self.cfg.dataset.map_bound.y[0]) / self.cfg.dataset.map_bound.y[2],
            (self.cfg.dataset.map_bound.z[1] - self.cfg.dataset.map_bound.z[0]) / self.cfg.dataset.map_bound.z[2],
        ]).astype(np.int16)

        pipe_param = {k: v for k, v in self.pipe_param.items() if v is not None}

        pipeline = self.pipe_cls.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            **pipe_param,
            unet=unet,
            safety_checker=None,
            feature_extractor=None,  # since v1.5 has default, we need to override
            torch_dtype=weight_dtype,
            grid_size=grid_size,
            use_cross_attn_cond=self.cfg.model.use_cross_attn_cond,
            use_map_cond=self.cfg.model.use_map_cond,
        )
        # NOTE: this scheduler does not take generator as kwargs.
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)

        if self.cfg.runner.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        image_logs = []
        progress_bar = tqdm(
            range(
                0,
                len(self.cfg.runner.validation_index)
                * self.cfg.runner.validation_times,
            ),
            desc="Val Steps",
        )

        # construc query
        coords, query = make_query(grid_size)

        for validation_i in self.cfg.runner.validation_index:
            raw_data = self.val_dataset[validation_i]  # cannot index loader
            val_input = collate_fn(
                [raw_data], self.cfg.dataset.template_t5, self.cfg.dataset.template_clip,
                is_train=False, bbox_mode=self.cfg.model.bbox_mode,
                bbox_view_shared=self.cfg.model.bbox_view_shared,
                map_cond_type=self.cfg.model.map_cond_type,
            )

            # let different prompts have the same random seed
            if self.cfg.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device).manual_seed(
                    self.cfg.seed
                )

            # for each input param, we generate several times to check variance.
            gen_occ_list, gen_latent_list, input_latent_list = [], [], []
            for _ in range(self.cfg.runner.validation_times):
                with torch.autocast("cuda"):
                    latents, input_latents, occupancy, voxcoord_with_occupancy = pipeline(
                        prompt=val_input["captions_t5"],
                        image=val_input["triplane_map"],
                        # latents=val_input["triplane"],
                        height=self.cfg.dataset.tri_size[0]+self.cfg.dataset.tri_size[2],   # X+Z 100+16
                        width=self.cfg.dataset.tri_size[1]+self.cfg.dataset.tri_size[2],    # Y+Z 100+16
                        query=query.to(val_input["triplane"].device),
                        coords=coords.to(val_input["triplane"].device),
                        generator=generator,
                        bev_controlnet_kwargs=val_input["kwargs"],
                        **self.cfg.runner.pipeline_param,
                    )

                progress_bar.update(1)

            # only save the late_time generation result
            occupancy_img = draw_occ_triplane(occupancy[0], show=False)
            latent_img = draw_latent_field2D(latents[0], show=False, title="Output")
            input_latent_img = draw_latent_field2D(input_latents[0], show=False, title="Input")
            gen_occ_list.append(to_pil_image(occupancy_img))
            gen_latent_list.append(to_pil_image(latent_img))
            input_latent_list.append(to_pil_image(input_latent_img))
    
            # save original input
            target_occ = to_pil_image(visualize_planemap(val_input["triplane_map"][0], show=False))
            target_latent = to_pil_image(draw_latent_field2D(val_input["triplane"][0], show=False, title="Target"))

            # bev_map
            map_img_np = visualize_map(
                self.cfg, val_input["bev_map_with_aux"][0])
            image_logs.append(
                {
                    "map_img_np": map_img_np,  # condition
                    "gen_occ_list": gen_occ_list,  # output
                    "gen_latent_list": gen_latent_list,  # output
                    "input_latent": input_latent_list,  # input
                    "target_occ": target_occ,  # target
                    "target_latent": target_latent,  # target
                    "validation_prompt": val_input["captions_clip"][0],
                }
            )

        for tracker in trackers:
            if tracker.name == "tensorboard":
                for log in image_logs:
                    map_img_np = log["map_img_np"]
                    validation_prompt = log["validation_prompt"]

                    formatted_images = format_ori_with_gen(
                        log["target_occ"], log["gen_occ_list"], nrow=2)
                    tracker.writer.add_image(
                        validation_prompt + " (occupancy)", formatted_images, step,
                        dataformats="HWC")

                    formatted_images = format_ori_with_gen(
                        log["target_latent"], log["gen_latent_list"], log["input_latent"], nrow=1)
                    tracker.writer.add_image(
                        validation_prompt + " (latent)", formatted_images,
                        step, dataformats="HWC")

                    tracker.writer.add_image(
                        "map: " + validation_prompt, map_img_np, step,
                        dataformats="HWC")
            elif tracker.name == "wandb":
                raise NotImplementedError("Do not use wandb.")
                formatted_images = []

                for log in image_logs:
                    images = log["images"]
                    validation_prompt = log["validation_prompt"]
                    validation_image = log["validation_image"]

                    formatted_images.append(
                        wandb.Image(
                            validation_image,
                            caption="Controlnet conditioning"))

                    for image in images:
                        image = wandb.Image(image, caption=validation_prompt)
                        formatted_images.append(image)

                tracker.log({"validation": formatted_images})
            else:
                logging.warn(
                    f"image logging not implemented for {tracker.name}")

        return image_logs
