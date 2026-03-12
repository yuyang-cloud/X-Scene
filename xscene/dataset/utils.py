from typing import Tuple, List
from functools import partial
import random

import torch
import numpy as np

from transformers import CLIPTokenizer, T5Tokenizer
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from ..runner.utils import trans_boxes_to_views
from ..misc.common import stack_tensors_in_dicts


META_KEY_LIST = [
    "gt_bboxes_3d",
    "gt_labels_3d",
    "camera_intrinsics",
    "camera2ego",
    "lidar2ego",
    "ego2global",
    "lidar2camera",
    "camera2lidar",
    "lidar2image",
    "img_aug_matrix",
    "metas",
]


def _tokenize_captions(examples, template_t5,  template_clip, tokenizer_t5=None, tokenizer_clip=None, is_train=True):
    results = {}
    captions_t5 = []
    captions_clip = []
    for example in examples:
        caption_t5 = template_t5.format(**example["metas"].data)
        caption_clip = template_clip.format(**example["metas"].data)
        captions_t5.append(caption_t5)
        captions_clip.append(caption_clip)
    captions_t5.append("")
    captions_clip.append("")
    results["captions_t5"] = captions_t5
    results["captions_clip"] = captions_clip

    # pad in the collate_fn function
    if tokenizer_t5 is not None:
        inputs = tokenizer_t5(
            captions_t5,
            max_length=tokenizer_t5.model_max_length,
            padding="do_not_pad",
            truncation=True,
        )
        input_ids = inputs.input_ids
        # pad to the longest of current batch (might differ between cards)
        padded_tokens = tokenizer_t5.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).input_ids
        results["padded_tokens_t5"] = padded_tokens
    
    if tokenizer_clip is not None:
        inputs = tokenizer_clip(
            captions_clip,
            max_length=tokenizer_clip.model_max_length,
            padding="do_not_pad",
            truncation=True,
        )
        input_ids = inputs.input_ids
        # pad to the longest of current batch (might differ between cards)
        padded_tokens = tokenizer_clip.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).input_ids
        results["padded_tokens_clip"] = padded_tokens

    return results


def ensure_canvas(coords, canvas_size: Tuple[int, int]):
    """Box with any point in range of canvas should be kept.

    Args:
        coords (_type_): _description_
        canvas_size (Tuple[int, int]): _description_

    Returns:
        np.array: mask on first axis.
    """
    (h, w) = canvas_size
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    w_mask = np.any(np.logical_and(
        coords[..., 0] > 0, coords[..., 0] < w), axis=1)
    h_mask = np.any(np.logical_and(
        coords[..., 1] > 0, coords[..., 1] < h), axis=1)
    c_mask = np.logical_and(c_mask, np.logical_and(w_mask, h_mask))
    return c_mask


def ensure_positive_z(coords):
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    return c_mask


def random_0_to_1(mask: np.array, num):
    assert mask.ndim == 1
    inds = np.where(mask == 0)[0].tolist()
    random.shuffle(inds)
    mask = np.copy(mask)
    mask[inds[:num]] = 1
    return mask


def _transform_all(examples, matrix_key, proj):
    """project all bbox to views, return 2d coordinates.

    Args:
        examples (List): collate_fn input.

    Returns:
        2-d list: List[List[np.array]] for B, N_cam. Can be [].
    """
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    # lidar2image (np.array): lidar to image view transformation
    trans_matrix = np.stack([example[matrix_key].data.numpy()
                            for example in examples], axis=0)
    # img_aug_matrix (np.array): augmentation matrix
    img_aug_matrix = np.stack([example['img_aug_matrix'].data.numpy()
                               for example in examples], axis=0)
    B, N_cam = trans_matrix.shape[:2]

    bboxes_coord = []
    # for each keyframe set
    for idx in range(B):
        # if zero, add empty list
        if len(gt_bboxes_3d[idx]) == 0:
            # keep N_cam dim for convenient
            bboxes_coord.append([None for _ in range(N_cam)])
            continue

        coords_list = trans_boxes_to_views(
            gt_bboxes_3d[idx], trans_matrix[idx], img_aug_matrix[idx], proj)
        bboxes_coord.append(coords_list)
    return bboxes_coord


def _preprocess_bbox(bbox_mode, canvas_size, examples, is_train=True,
                     view_shared=False, use_3d_filter=True, bbox_add_ratio=0,
                     bbox_add_num=0, bbox_drop_ratio=0, keyframe_rate=1):
    """Pre-processing for bbox
    .. code-block:: none

                                       up z
                        front x           ^
                             /            |
                            /             |
              (x1, y0, z1) + -----------  + (x1, y1, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y<-------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)

    Args:
        bbox_mode (str): type of bbox raw data.
            cxyz -> x1y1z1, x1y0z1, x1y1z0, x0y1z1;
            all-xyz -> all 8 corners xyz;
            owhr -> center, l, w, h, z-orientation.
        canvas_size (2-tuple): H, W of input images
        examples: collate_fn input
        view_shared: if enabled, all views share same set of bbox and output
            N_cam=1; otherwise, use projection to keep only visible bboxes.
    Return:
        in form of dict:
            bboxes (Tensor): B, N_cam, max_len, ...
            classes (LongTensor): B, N_cam, max_len
            masks: 1 for data, 0 for padding
    """
    # init data
    bboxes = []
    classes = []
    max_len = 0
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    gt_labels_3d: List[torch.Tensor] = [
        example["gt_labels_3d"].data for example in examples]

    # params
    B = len(gt_bboxes_3d)
    N_cam = len(examples[0]['lidar2image'].data.numpy())
    N_out = 1 if view_shared else N_cam

    bboxes_coord = None
    if not view_shared and not use_3d_filter:
        bboxes_coord = _transform_all(examples, 'lidar2image', True)
    elif not view_shared:
        bboxes_coord_3d = _transform_all(examples, 'lidar2camera', False)

    # for each keyframe set
    for idx in range(B):
        bboxes_kf = gt_bboxes_3d[idx]
        classes_kf = gt_labels_3d[idx]

        # if zero, add zero length tensor (for padding).
        if len(bboxes_kf) == 0:
            set_box_to_none = True
        elif idx % keyframe_rate != 0 and is_train:  # only for non-keyframes
            if random.random() < bbox_drop_ratio:
                set_box_to_none = True
            else:
                set_box_to_none = False
        else:
            set_box_to_none = False
        if set_box_to_none:
            bboxes.append([None] * N_out)
            classes.append([None] * N_out)
            continue

        # whether share the boxes across views, filtered by 2d projection.
        if not view_shared:
            index_list = []  # each view has a mask
            if use_3d_filter:
                coords_list = bboxes_coord_3d[idx]
                filter_func = ensure_positive_z
            else:
                # filter bbox according to 2d projection on image canvas
                coords_list = bboxes_coord[idx]
                # judge coord by cancas_size
                filter_func = partial(ensure_canvas, canvas_size=canvas_size)
            # we do not need to handle None since we already filter for len=0
            for coords in coords_list:
                c_mask = filter_func(coords)
                if random.random() < bbox_add_ratio and is_train:
                    c_mask = random_0_to_1(c_mask, bbox_add_num)
                index_list.append(c_mask)
                max_len = max(max_len, c_mask.sum())
        else:
            # we use as mask, torch.bool is important
            index_list = [torch.ones(len(bboxes_kf), dtype=torch.bool)]
            max_len = max(max_len, len(bboxes_kf))

        # construct data
        if bbox_mode == 'cxyz':
            # x1y1z1, x1y0z1, x1y1z0, x0y1z1
            bboxes_pt = bboxes_kf.corners[:, [6, 5, 7, 2]]
        elif bbox_mode == 'all-xyz':
            bboxes_pt = bboxes_kf.corners  # n x 8 x 3
        elif bbox_mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {bbox_mode}")
        bboxes.append([bboxes_pt[ind] for ind in index_list])
        classes.append([classes_kf[ind] for ind in index_list])
        bbox_shape = bboxes_pt.shape[1:]

    # there is no (visible) boxes in this batch
    if max_len == 0:
        return None, None

    # pad and construct mask
    # `bbox_shape` should be set correctly
    ret_dict = pad_bboxes_to_maxlen(
        [B, N_out, max_len, *bbox_shape], max_len, bboxes, classes)
    return ret_dict, bboxes_coord


def pad_bboxes_to_maxlen(
        bbox_shape, max_len, bboxes=None, classes=None, masks=None, **kwargs):
    B, N_out = bbox_shape[:2]
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape[3:])
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.long)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)
    if bboxes is not None:
        for _b in range(B):
            _bboxes = bboxes[_b]
            _classes = classes[_b]
            for _n in range(N_out):
                if _bboxes[_n] is None:
                    continue  # empty for this view
                this_box_num = len(_bboxes[_n])
                ret_bboxes[_b, _n, :this_box_num] = _bboxes[_n]
                ret_classes[_b, _n, :this_box_num] = _classes[_n]
                if masks is not None:
                    ret_masks[_b, _n, :this_box_num] = masks[_b, _n]
                else:
                    ret_masks[_b, _n, :this_box_num] = True

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict

def collate_fn(
    examples: Tuple[dict, ...],
    template_t5: str,
    template_clip: str,
    tokenizer_t5: T5Tokenizer = None,
    tokenizer_clip: CLIPTokenizer = None,
    is_train: bool = True,
    bbox_mode: str = None,
    bbox_view_shared: bool = False,
    bbox_drop_ratio: float = 0,
    bbox_add_ratio: float = 0,
    bbox_add_num: int = 3,
    map_cond_type: str = 'bev_seg',
    keyframe_rate: int = 1,
    ref_length: int = 0,
):
    """
    We need to handle:
    1. make multi-view images (img) into tensor -> [N, 6, 3, H, W]
    2. make triplane into tensor -> [N, C, X+Z, Y+Z]
    3. make masks (gt_masks_bev, gt_aux_bev) into tensor
        -> [N, 25 = 8 map + 10 obj + 7 aux, 200, 200]
    4. make masks (gt_masks_triplane) into tensor
        -> [N, 17, 216, 216]
    5. make caption (location, desctiption, timeofday) and tokenize, padding
        -> [N, pad_length]
    6. extract camera parameters (camera_intrinsics, camera2lidar)
        camera2lidar: A @ v_camera = v_lidar
        -> [N, 6, 3, 7]
    We keep other meta data as original.
    """
    if bbox_add_ratio > 0 and is_train:
        assert bbox_view_shared == False, "You cannot add any box on view shared."

    # video: ref + examples
    if ref_length > 0:
        ref_examples = examples[:ref_length]
        examples = examples[ref_length:]
        ref_values = torch.stack([example["img"].data for example in ref_examples])
        ref_values = ref_values.to(
            memory_format=torch.contiguous_format).float()
        if "can_bus" in examples[0]:
            ref_can_bus = torch.stack([example["can_bus"].data for example in ref_examples])
            ref_can_bus = ref_can_bus.to(
                memory_format=torch.contiguous_format).float()
            can_bus = torch.stack([example["can_bus"].data for example in examples])
            can_bus = can_bus.to(
                memory_format=torch.contiguous_format).float()

    # example images
    pixel_values = torch.stack([example["img"].data for example in examples])
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format).float()

    # camera param
    # TODO: camera2lidar should be changed to lidar2camera
    # fmt: off
    camera_param = torch.stack([torch.cat([
        example["camera_intrinsics"].data[:, :3, :3],  # 3x3 is enough
        example["camera2lidar"].data[:, :3],  # only first 3 rows meaningful
    ], dim=-1) for example in examples], dim=0)
    # fmt: on

    ret_dict = {
        "pixel_values": pixel_values,
        "camera_param": camera_param,
        "kwargs": {},
    }

    # ref_values and can_bus
    if ref_length > 0:
        ret_dict.update({"ref_values": ref_values})
        if "can_bus" in examples[0]:
            ret_dict.update({'can_bus': can_bus})
            ret_dict.update({"ref_can_bus": ref_can_bus})


    # bev_map_with_aux
    if "gt_aux_bev" in examples[0] and examples[0]["gt_aux_bev"] is not None:
        keys = ["gt_masks_bev", "gt_aux_bev"]
        assert bbox_drop_ratio == 0, "map is not affected in bbox_drop"
    else:
        keys = ["gt_masks_bev"]
    # fmt: off
    bev_map_with_aux = torch.stack([torch.from_numpy(np.concatenate([
        example[key] for key in keys  # np array, channel-last
    ], axis=0)).float() for example in examples], dim=0)  # float32
    # fmt: on
    ret_dict.update({
        "bev_map_with_aux": bev_map_with_aux,
    })


    if "triplane" in examples[0]:
        # triplane
        triplane = torch.stack(
            [torch.from_numpy(example["triplane"]) for example in examples])
        triplane = triplane.to(
            memory_format=torch.contiguous_format).float()

        # triplane condition map
        H, W = bev_map_with_aux.shape[-2:]
        if map_cond_type == 'bev_seg':      # choice_1: dense_mask_map: [12, 200, 200]
            new_map = torch.cat([
                bev_map_with_aux[:, 8:18, ...], # barrier, bicycle, bus, etc.
                bev_map_with_aux[:, 0:1, ...],  # driveable_area(driveable_surface)
                bev_map_with_aux[:, 2:3, ...],  # walk_way(sidewalk)
            ], dim=1)
        elif map_cond_type == 'hd_map':     # choice_2: sparse_vector_map: [3(road_vector)+10(obj_mask), 200, 200]
            bev_hdmap_w_box = torch.stack(
                [torch.from_numpy(example["bev_hdmap_w_box"]) for example in examples]) # 13, 200, 200
            new_map = bev_hdmap_w_box.to(
                memory_format=torch.contiguous_format).float()
        else:
            raise ValueError(f"Unknown map_cond_type {map_cond_type}")
        new_map = torch.rot90(new_map, k=-1, dims=(2, 3))    # lidar_coord -> ego_coord
        triplane_map = torch.zeros([new_map.shape[0], new_map.shape[1], 216, 216])
        triplane_map[:, :, :H, :W] = new_map

        ret_dict.update({
            "triplane": triplane,
            "triplane_map": triplane_map,
        })


    # layout_canvas and bev_hdmap
    if 'layout_canvas' in examples[0] and examples[0]['layout_canvas'] is not None:
        layout_canvas = torch.stack(
            [torch.from_numpy(example["layout_canvas"]) for example in examples])
        layout_canvas = layout_canvas.to(
            memory_format=torch.contiguous_format).float()
        ret_dict["layout_canvas"] = layout_canvas

        bev_hdmap = torch.stack(
            [torch.from_numpy(example["bev_hdmap"]) for example in examples])
        bev_hdmap = bev_hdmap.to(
            memory_format=torch.contiguous_format).float()
        ret_dict["bev_hdmap"] = bev_hdmap

    # occ_render_map
    if 'occ_render_image' in examples[0] and examples[0]['occ_render_image'] is not None:
        occ_render_image = torch.stack(
            [example["occ_render_image"] for example in examples])
        occ_render_image = occ_render_image.to(
            memory_format=torch.contiguous_format).float()
        ret_dict["occ_render_image"] = occ_render_image

        occ_render_depth = torch.stack(
            [example["occ_render_depth"] for example in examples])
        occ_render_depth = occ_render_depth.to(
            memory_format=torch.contiguous_format).float()
        ret_dict["occ_render_depth"] = occ_render_depth

    # bboxes_3d, convert to tensor
    # here we consider:
    # 1. do we need to filter bboxes for each view? use `view_shared`
    # 2. padding for one batch of data if need (with zero), and output mask.
    # 3. what is the expected output format? dict of kwargs to bbox embedder
    canvas_size = pixel_values.shape[-2:]
    if bbox_mode is not None:
        # NOTE: both can be None
        bboxes_3d_input, bbox_view_coord = _preprocess_bbox(
            bbox_mode, canvas_size, examples, is_train=is_train,
            view_shared=bbox_view_shared, bbox_add_ratio=bbox_add_ratio,
            bbox_add_num=bbox_add_num, bbox_drop_ratio=bbox_drop_ratio,
            keyframe_rate=keyframe_rate)
        if bboxes_3d_input is not None:
            bboxes_3d_input["cam_params"] = camera_param
        ret_dict["kwargs"]["bboxes_3d_data"] = bboxes_3d_input
    else:
        bbox_view_coord = None

    # captions: one real caption with one null caption
    caption_token_dict = _tokenize_captions(
        examples, template_t5, template_clip, tokenizer_t5, tokenizer_clip, is_train)
    ret_dict["captions_clip"] = caption_token_dict["captions_clip"][:-1]  # list of str
    ret_dict["captions_t5"] = caption_token_dict["captions_t5"][:-1]  # list of str
    if tokenizer_t5 is not None:
        # real captions in head; the last one is null caption
        # we omit "attention_mask": padded_tokens.attention_mask, seems useless
        ret_dict["input_ids_t5"] = caption_token_dict["padded_tokens_t5"][:-1]
        ret_dict["uncond_ids_t5"] = caption_token_dict["padded_tokens_t5"][-1:]
    if tokenizer_clip is not None:
        ret_dict["input_ids_clip"] = caption_token_dict["padded_tokens_clip"][:-1]
        ret_dict["uncond_ids_clip"] = caption_token_dict["padded_tokens_clip"][-1:]

    # other meta data
    meta_list_dict = dict()
    for key in META_KEY_LIST:
        try:
            meta_list = [example[key] for example in examples]
            meta_list_dict[key] = meta_list
        except KeyError:
            continue
    ret_dict['meta_data'] = meta_list_dict

    return ret_dict


def collate_t_fn(
    examples: Tuple[dict, ...],
    template_t5: str,
    template_clip: str,
    tokenizer_clip: CLIPTokenizer = None,
    tokenizer_t5: T5Tokenizer = None,
    **kwargs,
):
    ret_dicts = []
    bbox_maxlen = 0
    input_ids_t5_max_len = 0
    input_ids_clip_max_len = 0
    for example_ti in examples:
        ret_dict = collate_fn(
            example_ti, template_t5=template_t5, template_clip=template_clip,
            tokenizer_t5=tokenizer_t5, tokenizer_clip=tokenizer_clip, **kwargs)
        if ret_dict['kwargs']['bboxes_3d_data'] is not None:
            bb_shape = ret_dict['kwargs']['bboxes_3d_data']['bboxes'].shape
            bbox_maxlen = max(bbox_maxlen, bb_shape[2])
        if "input_ids_t5" in ret_dict:
            input_ids_t5_max_len = max(
                input_ids_t5_max_len, ret_dict['input_ids_t5'].shape[1])
        if "input_ids_clip" in ret_dict:
            input_ids_clip_max_len = max(
                input_ids_clip_max_len, ret_dict['input_ids_clip'].shape[1])
        ret_dicts.append(ret_dict)

    if bbox_maxlen != 0:
        for ret_dict in ret_dicts:
            bboxes_3d_data = ret_dict['kwargs']['bboxes_3d_data']
            # if it is None while others not, we replace it will all padding.
            bboxes_3d_data = {} if bboxes_3d_data is None else bboxes_3d_data
            new_data = pad_bboxes_to_maxlen(
                bb_shape, bbox_maxlen, **bboxes_3d_data)
            ret_dict['kwargs']['bboxes_3d_data'].update(new_data)

    def pad_input_ids(input_ids, input_ids_max_len, tokenizer):
        padded_tokens = tokenizer.pad(
            {"input_ids": input_ids}, padding="max_length",
            max_length=input_ids_max_len, return_tensors="pt",
        ).input_ids
        return padded_tokens

    if input_ids_t5_max_len != 0 or input_ids_clip_max_len != 0:
        for ret_dict in ret_dicts:
            if input_ids_t5_max_len != 0:
                ret_dict["input_ids_t5"] = pad_input_ids(ret_dict["input_ids_t5"],
                                                        input_ids_t5_max_len, tokenizer_t5)
                ret_dict["uncond_ids_t5"] = pad_input_ids(ret_dict["uncond_ids_t5"],
                                                        input_ids_t5_max_len, tokenizer_t5)
            if input_ids_clip_max_len != 0:
                ret_dict["input_ids_clip"] = pad_input_ids(ret_dict["input_ids_clip"],
                                                        input_ids_clip_max_len, tokenizer_clip)
                ret_dict["uncond_ids_clip"] = pad_input_ids(ret_dict["uncond_ids_clip"],
                                                        input_ids_clip_max_len, tokenizer_clip)

    # each example_ti have frame_len dim, we need to add batch dim.
    ret_dicts = stack_tensors_in_dicts(ret_dicts, dim=0)
    return ret_dicts
