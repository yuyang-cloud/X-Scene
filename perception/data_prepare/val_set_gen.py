"""Run generation on val set for testing.
"""

import os
import sys
import json
import copy
import hydra
from hydra.core.hydra_config import HydraConfig
import shutil
import logging
from glob import glob
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np

import torch
import torchvision
from torchvision.transforms import InterpolationMode
from accelerate import Accelerator

sys.path.append(".")
from perception.common.ddp_utils import concat_from_everyone
from xscene.misc.test_utils import (
    prepare_all, run_one_batch_img, run_one_batch_occ, update_progress_bar_config,
)


def copy_save_image(tmp, cfg, gen_imgs_list, post_trans):
    tmp_all = []
    for bi, template in enumerate(tmp): # bs
        for gen_id, gen_imgs in enumerate(gen_imgs_list[bi]):   # validation_times
            # for one generation with 6 views
            for idx, view in enumerate(cfg.dataset.view_order): # views
                # get index in label file
                filename = os.path.basename(template['filename'][idx])
                filename = Path(view) / f"_gen_{gen_id}".join(
                    os.path.splitext(filename))
                # save to path
                save_name = os.path.join(cfg.fid.img_gen_dir, filename)
                post_trans(gen_imgs[idx]).save(save_name)
            tmp_all.append(copy.deepcopy(template))
    return tmp_all

def copy_save_occ(tmp, cfg, gen_occ_list):
    tmp_all = []
    for bi, template in enumerate(tmp): # bs
        for gen_id, gen_occ in enumerate(gen_occ_list[bi]): # validation_times
            token_name, lidar_token, timestamp = template['token'], template['lidar_token'], template['timestamp']
            save_name = os.path.join(cfg.fid.img_gen_dir, f'{timestamp}_{token_name}.label')
            gen_occ = gen_occ.astype(np.uint8)
            gen_occ.tofile(save_name)
            tmp_all.append(copy.deepcopy(template))
    return tmp_all


def filter_tokens(meta_list, token_set):
    to_add_tmp = []
    for meta in meta_list:
        if meta['token'] in token_set:
            continue
        else:
            to_add_tmp.append(meta)
            token_set.add(meta['token'])
    return to_add_tmp, token_set


@hydra.main(version_base=None, config_path="../../configs", config_name=None)
def main(cfg):
    logging.info(
        f"Your config for fid:\n" + OmegaConf.to_yaml(cfg.fid, resolve=True))

    # pipeline and dataloader
    # this function also set global seed in cfg
    accelerator = Accelerator(
        mixed_precision=cfg.accelerator.mixed_precision,
        project_dir=HydraConfig.get().runtime.output_dir,
    )
    pipe, val_dataloader, weight_dtype = prepare_all(
        cfg, device=accelerator.device)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.log_root, "run_config.yaml"))
    if cfg.model.name == "img_unet":
        pipe.enable_vae_slicing()
    val_dataloader = accelerator.prepare(val_dataloader)
    pipe.to(accelerator.device)

    # random states
    if cfg.runner.validation_seed_global:
        global_generator = torch.manual_seed(
            cfg.seed + accelerator.process_index)
    else:
        global_generator = None

    # prepare
    generated_token = []

    # check resume
    if os.path.exists(cfg.fid.img_gen_dir) and cfg.model.name == "occ_unet":
        logging.info(
            f"Previous results exists: {cfg.fid.img_gen_dir}."
            f"Resume from them")
    elif os.path.exists(cfg.fid.img_gen_dir) and cfg.model.name == "img_unet":
        logging.info(
            f"Previous results exists: {cfg.fid.img_gen_dir}."
            f"Resume from them")
    else:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if cfg.model.name == "img_unet":
                for view in cfg.dataset.view_order:
                    os.makedirs(Path(cfg.fid.img_gen_dir) / view)
            elif cfg.model.name == "occ_unet":
                os.makedirs(cfg.fid.img_gen_dir)

    # post process
    if cfg.fid.raw_output:
        post_trans = []
    else:
        post_trans = [
            torchvision.transforms.Resize(
                OmegaConf.to_container(cfg.fid.resize, resolve=True),
                interpolation=InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.Pad(
                OmegaConf.to_container(cfg.fid.padding, resolve=True)
            ),
        ]
    post_trans = torchvision.transforms.Compose(post_trans)
    logging.info(f"Using post process: {post_trans}")

    # tqdm bar
    progress_bar = tqdm(
        range(len(val_dataloader)), desc="Steps", ncols=80,
        disable=not accelerator.is_main_process)
    update_progress_bar_config(
        pipe, ncols=80, disable=not accelerator.is_main_process)

    # run
    token_set = set()
    for val_input in val_dataloader:
        bs = len(val_input['meta_data']['metas'])
        accelerator.wait_for_everyone()

        if cfg.model.name == "img_unet":
            # now make labels
            tmp = []
            result_exist_list = []
            for bi in range(bs):
                # for one data item, we may generate several times, they
                # share label files.
                token = val_input['meta_data']['metas'][bi].data['token']
                filename = val_input['meta_data']['metas'][bi].data['filename']
                tmp.append({
                    "token": token,
                    "filename": filename,
                })
                cur_result_exist = []
                for file in filename:
                    rel_path = file.split('samples/')[-1]
                    folder, path = os.path.split(rel_path)
                    name, ext = os.path.splitext(path)
                    new_filename = f"{name}_gen_0{ext}"
                    new_file = os.path.join(cfg.fid.img_gen_dir, folder, new_filename)
                    cur_result_exist.append(os.path.exists(new_file))
                # all cameras should exist
                result_exist_list.append(all(cur_result_exist))
            assert len(result_exist_list) == len(tmp) == bs

            # check if the label file exists
            if not all(result_exist_list):
                # this function also set seed to as cfg
                map_img, ori_imgs, ori_imgs_wb, ori_imgs_wl, \
                    gen_imgs_list, gen_imgs_wb_list, gen_imgs_wl_list = run_one_batch_img(
                        cfg, pipe, val_input, weight_dtype,
                        global_generator=global_generator)

                # collect and save images on main process only
                if accelerator.num_processes > int(os.environ.get("LOCAL_WORLD_SIZE", accelerator.num_processes)):
                    # on multi-node, we first gather data, then save on disk.
                    tmp = concat_from_everyone(accelerator, tmp)
                    gen_imgs_list = concat_from_everyone(accelerator, gen_imgs_list)
                    if accelerator.is_main_process:
                        tmp = copy_save_image(tmp, cfg, gen_imgs_list, post_trans)
                    else:
                        pass
                else:
                    # on single-node, we save on disk, then gather label
                    tmp = copy_save_image(tmp, cfg, gen_imgs_list, post_trans)
                    tmp = concat_from_everyone(accelerator, tmp)
            else:
                # no need to run, just gather
                tmp = concat_from_everyone(accelerator, tmp)


        elif cfg.model.name == "occ_unet":
            # now make labels
            tmp = []
            result_exist_list = []
            for bi in range(bs):
                # for one data item, we may generate several times, they
                # share label files.
                token = val_input['meta_data']['metas'][bi].data['token']
                lidar_token = val_input['meta_data']['metas'][bi].data['lidar_token']
                timestamp = val_input['meta_data']['metas'][bi].data['timestamp']
                tmp.append({
                    "token": token,
                    "lidar_token": lidar_token,
                    "timestamp": timestamp,
                })
                if os.path.exists(os.path.join(cfg.fid.img_gen_dir, f"{timestamp}_{token}.label")):
                    result_exist_list.append(True)
                else:
                    result_exist_list.append(False)
            assert len(result_exist_list) == len(tmp) == bs

            # check if the label file exists
            if not all(result_exist_list):
                # run one batch
                gen_occ_list, gen_coord_occ_list, target_occ_list, \
                    map_imgs, vis_target_occ_list, vis_target_latent_list, vis_gen_occ_list, vis_gen_latent_list \
                        = run_one_batch_occ(cfg, pipe, val_input, weight_dtype)
                
                # collect and save images on main process only
                if accelerator.num_processes > int(os.environ.get("LOCAL_WORLD_SIZE", accelerator.num_processes)):
                    # on multi-node, we first gather data, then save on disk.
                    tmp = concat_from_everyone(accelerator, tmp)
                    gen_occ_list = concat_from_everyone(accelerator, gen_occ_list)
                    if accelerator.is_main_process:
                        tmp = copy_save_occ(tmp, cfg, gen_occ_list)
                    else:
                        pass
                else:
                    # on single-node, we save on disk, then gather label
                    tmp = copy_save_occ(tmp, cfg, gen_occ_list)
                    tmp = concat_from_everyone(accelerator, tmp)
            else:
                # no need to run, just gather
                tmp = concat_from_everyone(accelerator, tmp)

        accelerator.wait_for_everyone()

        # main process construct data.
        if accelerator.is_main_process:
            tmp, token_set = filter_tokens(tmp, token_set)
        # update bar
        progress_bar.update(1)

    # end
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
