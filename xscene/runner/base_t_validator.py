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
    img_concat_v,
)
from xscene.runner.base_validator import BaseValidator
from xscene.misc.common import move_to
from xscene.misc.test_utils import draw_box_on_imgs, draw_lane_on_imgs
from xscene.pipeline.pipeline_bev_controlnet import (
    BEVStableDiffusionPipelineOutput,
)
from xscene.dataset.utils import collate_fn
from xscene.networks.unet_addon_rawbox import BEVControlNetModel


def format_ori_with_gen(ori_img, gen_img_list):
    formatted_images = []

    # first image is input, followed by generations.
    formatted_images.append(np.asarray(ori_img))

    for image in gen_img_list:
        formatted_images.append(np.asarray(image))

    # formatted_images = np.stack(formatted_images)
    # 0-255 np -> 0-1 tensor -> grid -> 0-255 pil -> np
    formatted_images = torchvision.utils.make_grid(
        [to_tensor(im) for im in formatted_images], nrow=1)
    formatted_images = np.asarray(
        to_pil_image(formatted_images))
    return formatted_images


class BaseTValidator(BaseValidator):
    def construct_visual(self, images_list, val_input, with_box, with_lane):
        frame_list = []
        frame_list_wb = []
        frame_list_wl = []
        for idx, framei in enumerate(images_list):
            frame = concat_6_views(framei, oneline=True)
            if with_box:
                frame_with_box = concat_6_views(
                    draw_box_on_imgs(
                        self.cfg, idx, val_input, framei),
                    oneline=True)
            if with_lane:
                frame_with_lane = concat_6_views(
                draw_lane_on_imgs(
                    framei, val_input["layout_canvas"][idx])
                )
            frame_list.append(frame)
            frame_list_wb.append(frame_with_box)
            frame_list_wl.append(frame_with_lane)
        frames = img_concat_v(*frame_list)
        if with_box:
            frames_wb = img_concat_v(*frame_list_wb)
        else:
            frames_wb = None
        if with_lane:
            frames_wl = img_concat_v(*frame_list_wl)
        else:
            frames_wl = None
        return frames, frames_wb, frames_wl

    def prepare_pipeline(self, controlnet, unet, scene_embedder, 
                         can_bus_embedder, weight_dtype, device):
        controlnet.eval()  # important !!!
        unet.eval()
        scene_embedder.eval()
        can_bus_embedder.eval()

        pipeline = self.pipe_cls.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            **self.pipe_param,
            unet=unet,
            controlnet=controlnet,
            scene_embedder=scene_embedder,
            can_bus_embedder=can_bus_embedder,
            safety_checker=None,
            feature_extractor=None,  # since v1.5 has default, we need to override
            torch_dtype=weight_dtype,
        )
        # NOTE: this scheduler does not take generator as kwargs.
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)

        if self.cfg.runner.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()
        return pipeline

    def validate(
        self,
        controlnet: BEVControlNetModel,
        unet,
        scene_embedder,
        can_bus_embedder,
        trackers: Tuple[GeneralTracker, ...],
        step, weight_dtype, device
    ):
        logging.info(f"[{self.__class__.__name__}] Running validation... ")
        torch.cuda.empty_cache()

        pipeline = self.prepare_pipeline(
            controlnet, unet, scene_embedder, can_bus_embedder, weight_dtype, device)

        image_logs = []
        total_run_times = len(
            self.cfg.runner.validation_index) * self.cfg.runner.validation_times
        if self.cfg.runner.pipeline_param['init_noise'] == 'both':
            total_run_times *= 2
        progress_bar = tqdm(
            range(0, total_run_times), desc="Val Steps")

        for validation_i in self.cfg.runner.validation_index:
            raw_data = self.val_dataset[validation_i]  # cannot index loader
            val_input = collate_fn(
                raw_data, self.cfg.dataset.template_t5, self.cfg.dataset.template_clip,
                is_train=False, bbox_mode=self.cfg.model.bbox_mode,
                bbox_view_shared=self.cfg.model.bbox_view_shared,
                ref_length=self.cfg.model.ref_length,
            )
            # camera_emb = self._embed_camera(val_input["camera_param"])
            camera_param = val_input["camera_param"].to(weight_dtype)

            ref_image = val_input["ref_values"].to(weight_dtype)
            ref_can_bus = val_input["ref_can_bus"].to(weight_dtype)
            can_bus = val_input["can_bus"].to(weight_dtype)
            can_bus = torch.cat([ref_can_bus, can_bus], dim=0)

            controlnet_image = val_input["bev_map_with_aux"].to(
                dtype=weight_dtype)
            controlnet_image = [controlnet_image]
            # layout canvas
            if 'with_layout_canvas' in self.cfg.model.controlnet and self.cfg.model.controlnet.with_layout_canvas:
                bev_hdmap = val_input["bev_hdmap"].to(dtype=weight_dtype)
                layout_canvas = val_input["layout_canvas"].to(dtype=weight_dtype)
                controlnet_image += [bev_hdmap, layout_canvas]
            if 'with_occ_render_img' in self.cfg.model.controlnet and self.cfg.model.controlnet.with_occ_render_img:
                occ_render_img = val_input["occ_render_image"].to(dtype=weight_dtype)
                occ_render_depth = val_input["occ_render_depth"].to(dtype=weight_dtype)
                controlnet_image += [occ_render_img, occ_render_depth]

            # let different prompts have the same random seed
            if self.cfg.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device).manual_seed(
                    self.cfg.seed
                )

            def run_once(pipe_param):
                for _ in range(self.cfg.runner.validation_times):
                    with torch.autocast("cuda"):
                        image: BEVStableDiffusionPipelineOutput = pipeline(
                            prompt=val_input["captions_clip"],
                            prompt_t5=val_input["captions_t5"],
                            image=controlnet_image,
                            ref_image=ref_image,
                            can_bus=can_bus,
                            camera_param=camera_param,
                            height=self.cfg.dataset.image_size[0],
                            width=self.cfg.dataset.image_size[1],
                            generator=generator,
                            bev_controlnet_kwargs=val_input["kwargs"],
                            img_metas=val_input['meta_data'],
                            **pipe_param,
                        )
                    gen_frames, gen_frames_wb, gen_frames_wl = self.construct_visual(
                        image.images, val_input,
                        self.cfg.runner.validation_show_box,
                        self.cfg.runner.validation_show_line)
                    gen_list.append(gen_frames)
                    if self.cfg.runner.validation_show_box:
                        gen_wb_list.append(gen_frames_wb)
                    if self.cfg.runner.validation_show_line:
                        gen_wl_list.append(gen_frames_wl)

                    progress_bar.update(1)
                
            # for each input param, we generate several times to check variance.
            gen_list, gen_wb_list, gen_wl_list = [], [], []
            pipeline_param = {
                k: v for k, v in self.cfg.runner.pipeline_param.items()}
            if self.cfg.runner.pipeline_param['init_noise'] != 'both':
                run_once(pipeline_param)
            else:
                pipeline_param['init_noise'] = "same"
                run_once(pipeline_param)
                pipeline_param['init_noise'] = "rand"
                run_once(pipeline_param)

            # make image for 6 views and save to dict
            ori_imgs = [[
                to_pil_image(img_m11_to_01(val_input["pixel_values"][j][i]))
                for i in range(6)
            ] for j in range(self.cfg.model.video_length)]
            ori_img, ori_img_wb, ori_img_wl = self.construct_visual(
                ori_imgs, val_input, True, True)
            map_img_np = visualize_map(
                self.cfg, val_input["bev_map_with_aux"][0])
            image_logs.append(
                {
                    "map_img_np": map_img_np,  # condition
                    "gen_img_list": gen_list,  # output
                    "gen_img_wb_list": gen_wb_list,  # output
                    "gen_img_wl_list": gen_wl_list,  # output
                    "ori_img": ori_img,  # input
                    "ori_img_wb": ori_img_wb,  # input
                    "ori_img_wl": ori_img_wl,  # input
                    "validation_prompt": val_input["captions_clip"][0],
                }
            )

        for tracker in trackers:
            if tracker.name == "tensorboard":
                for log in image_logs:
                    map_img_np = log["map_img_np"]
                    validation_prompt = log["validation_prompt"]

                    formatted_images = format_ori_with_gen(
                        log["ori_img"], log["gen_img_list"])
                    tracker.writer.add_image(
                        validation_prompt, formatted_images, step,
                        dataformats="HWC")

                    formatted_images = format_ori_with_gen(
                        log["ori_img_wb"], log["gen_img_wb_list"])
                    tracker.writer.add_image(
                        validation_prompt + " (with box)", formatted_images,
                        step, dataformats="HWC")

                    formatted_images = format_ori_with_gen(
                        log["ori_img_wl"], log["gen_img_wl_list"])
                    tracker.writer.add_image(
                        validation_prompt + " (with lane)", formatted_images,
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
