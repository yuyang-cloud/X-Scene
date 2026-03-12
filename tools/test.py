import os
import sys
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

sys.path.append(".")  # noqa
from xscene.runner.utils import concat_6_views
from xscene.misc.test_utils import (
    prepare_all, run_one_batch_img, run_one_batch_occ
)
from xscene.runner.occ_validator import format_ori_with_gen

transparent_bg = False
target_map_size = 400
# target_map_size = 800


def output_func(x): return concat_6_views(x)
# def output_func(x): return concat_6_views(x, oneline=True)
# def output_func(x): return img_concat_h(*x[:3])


@hydra.main(version_base=None, config_path="../configs", config_name=None)
def main(cfg: DictConfig):
    if cfg.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')

    output_dir = to_absolute_path(cfg.resume_from_checkpoint)
    original_overrides = OmegaConf.load(
        os.path.join(output_dir, "hydra/overrides.yaml"))
    current_overrides = HydraConfig.get().overrides.task

    # getting the config name of this job.
    config_name = HydraConfig.get().job.config_name
    # concatenating the original overrides with the current overrides
    overrides = original_overrides + current_overrides
    # compose a new config from scratch
    cfg = hydra.compose(config_name, overrides=overrides)
    logging.info(f"Your validation index: {cfg.runner.validation_index}")

    #### setup everything ####
    pipe, val_dataloader, weight_dtype = prepare_all(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.log_root, "run_config.yaml"))

    #### start ####
    total_num = 0
    progress_bar = tqdm(
        range(len(val_dataloader) * cfg.runner.validation_times),
        desc="Steps",
    )
    for val_input in val_dataloader:
        if cfg.model.name == "img_unet":
            return_tuples = run_one_batch_img(cfg, pipe, val_input, weight_dtype,
                                        transparent_bg=transparent_bg,
                                        map_size=target_map_size)
            # save result
            for map_img, ori_imgs, ori_imgs_wb, ori_imgs_wl, \
                gen_imgs_list, gen_imgs_wb_list, gen_imgs_wl_list in zip(*return_tuples):
                # save map
                map_img.save(os.path.join(cfg.log_root, f"{total_num}_map.png"))

                # save img
                ori_img = output_func(ori_imgs)
                gen_imgs_list = [output_func(gen_imgs) for gen_imgs in gen_imgs_list]
                img = format_ori_with_gen(ori_img, gen_imgs_list, nrow=1, to_array=False)
                img.save(os.path.join(cfg.log_root, f"{total_num}_img.png"))

                # save img with box
                ori_img_with_box = output_func(ori_imgs_wb)
                gen_imgs_wb_list = [output_func(gen_imgs_wb) for gen_imgs_wb in gen_imgs_wb_list]
                img_with_box = format_ori_with_gen(ori_img_with_box, gen_imgs_wb_list, nrow=1, to_array=False)
                img_with_box.save(os.path.join(cfg.log_root, f"{total_num}_img_box.png"))

                # save img with lane
                ori_img_with_lane = output_func(ori_imgs_wl)
                gen_imgs_wl_list = [output_func(gen_imgs_wl) for gen_imgs_wl in gen_imgs_wl_list]
                img_with_lane = format_ori_with_gen(ori_img_with_lane, gen_imgs_wl_list, nrow=1, to_array=False)
                img_with_lane.save(os.path.join(cfg.log_root, f"{total_num}_img_lane.png"))

                total_num += 1


        elif cfg.model.name == "occ_unet":
            return_tuples = run_one_batch_occ(cfg, pipe, val_input, weight_dtype,
                                            map_size=target_map_size, vis=True)
            # save result
            for i, (gen_occ_list, gen_coord_occ_list, target_occ_list, \
                map_img, tgt_occ, tgt_latent, gen_occ, gen_latent) in enumerate(zip(*return_tuples)):

                # save map
                map_img.save(os.path.join(cfg.log_root, f"{total_num}_map.png"))

                # save occ
                occ_img = format_ori_with_gen(tgt_occ[0], gen_occ, nrow=2, to_array=False)
                occ_img.save(os.path.join(cfg.log_root, f"{total_num}_occ.png"))
                # save latent
                latent_img = format_ori_with_gen(tgt_latent[0], gen_latent, nrow=1, to_array=False)
                latent_img.save(os.path.join(cfg.log_root, f"{total_num}_latent.png"))

                total_num += 1

        # update bar
        progress_bar.update(cfg.runner.validation_times)


if __name__ == "__main__":
    main()
