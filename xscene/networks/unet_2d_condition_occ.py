# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import copy
import random
from einops import repeat, rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from diffusers.configuration_utils import register_to_config
from diffusers.models.unet_2d_condition import (
    UNet2DConditionModel,
    UNet2DConditionOutput,
)
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UpBlock2D,
)
from diffusers.models.resnet import (
    Downsample2D,
    Upsample2D,
    ResnetBlock2D,
)
from diffusers.models.transformer_2d import Transformer2DModel
from ..misc.common import _get_module, _set_module
from .blocks import (
    TriplaneConv,
    TriplaneNorm,
    TriplaneSiLU,
    TriplaneDownsample2x,
    TriplaneUpsample2x,
    TriplaneResBlock,
    TriplaneTransformerModel,
)
from .map_embedder import BEVControlNetConditioningEmbedding, TriplaneMapConditioningEmbedding
from ..misc.common import load_module


class UNet2DConditionModelOcc(UNet2DConditionModel):
    r"""
    UNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2DCrossAttn"`):
            The mid block type. Choose from `UNetMidBlock2DCrossAttn` or `UNetMidBlock2DSimpleCrossAttn`, will skip the
            mid block layer if `None`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use.
        only_cross_attention(`bool` or `Tuple[bool]`, *optional*, default to `False`):
            Whether to include self-attention in the basic transformer blocks, see
            [`~models.attention.BasicTransformerBlock`].
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, it will skip the normalization and activation layers in post-processing
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If given, `encoder_hidden_states` will be projected from this dimension to `cross_attention_dim`.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for resnet blocks, see [`~models.resnet.ResnetBlock2D`]. Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to None):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to None):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        num_class_embeds (`int`, *optional*, defaults to None):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        time_embedding_type (`str`, *optional*, default to `positional`):
            The type of position embedding to use for timesteps. Choose from `positional` or `fourier`.
        time_embedding_dim (`int`, *optional*, default to `None`):
            An optional override for the dimension of the projected time embedding.
        time_embedding_act_fn (`str`, *optional*, default to `None`):
            Optional activation function to use on the time embeddings only one time before they as passed to the rest
            of the unet. Choose from `silu`, `mish`, `gelu`, and `swish`.
        timestep_post_act (`str, *optional*, default to `None`):
            The second activation function to use in timestep embedding. Choose from `silu`, `mish` and `gelu`.
        time_cond_proj_dim (`int`, *optional*, default to `None`):
            The dimension of `cond_proj` layer in timestep embedding.
        conv_in_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_in` layer.
        conv_out_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_out` layer.
        projection_class_embeddings_input_dim (`int`, *optional*): The dimension of the `class_labels` input when
            using the "projection" `class_embed_type`. Required when using the "projection" `class_embed_type`.
        class_embeddings_concat (`bool`, *optional*, defaults to `False`): Whether to concatenate the time
            embeddings with the class embeddings.
        mid_block_only_cross_attention (`bool`, *optional*, defaults to `None`):
            Whether to use cross attention with the mid block when using the `UNetMidBlock2DSimpleCrossAttn`. If
            `only_cross_attention` is given as a single boolean and `mid_block_only_cross_attention` is None, the
            `only_cross_attention` value will be used as the value for `mid_block_only_cross_attention`. Else, it will
            default to `False`.
    """

    _supports_gradient_checkpointing = True
    _WARN_ONCE = 0

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,
        # parameter added, we should keep all above (do not use kwargs)
        tri_size: Tuple[int, int, int] = (100, 100, 16),
        is_rollout: bool = True,
        conv_down: bool = True,
        conv_up: bool = True,
        tri_z_down: bool = False,
        # map condition params
        use_map_cond: bool = True,
        map_embedder_cls: str = None,
        map_embedder_param: dict = None,
        map_cond_type: str = "bev_seg",
        map_size_bev_seg: Tuple[int, int, int] = None,
        map_size_hd_map: Tuple[int, int ,int] = None,
        conditioning_embedding_out_channels: Optional[Tuple[int]] = None,
        use_uncond_map: str = None,
        drop_cond_ratio: float = 0.0,
        # box condition params
        use_cross_attn_cond: bool = True,
        bbox_embedder_cls: str = None,
        bbox_embedder_param: dict = None,
    ):
        super().__init__(
            sample_size=sample_size, in_channels=in_channels,
            out_channels=out_channels, center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos, freq_shift=freq_shift,
            down_block_types=down_block_types, mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor, act_fn=act_fn,
            norm_num_groups=norm_num_groups, norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel, conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,)

        self.img_size = [int(s) for s in tri_size] \
            if tri_size is not None else None

        # replace modules
        replace_modules = []
        for name, mod in self.named_modules():
            if isinstance(mod, (Transformer2DModel, ResnetBlock2D, Downsample2D, Upsample2D)):
                replace_modules.append(name)
            elif isinstance(mod, (nn.Conv2d, nn.GroupNorm, nn.SiLU)) and 'time' not in name:
                if not any(replace_name in name for replace_name in replace_modules):
                    replace_modules.append(name)

        for name, mod in list(self.named_modules()):
            if name in replace_modules:
                if isinstance(mod, Transformer2DModel):
                    _set_module(self, name, TriplaneTransformerModel(
                        num_attention_heads=mod.num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        in_channels=mod.in_channels,
                        norm_num_groups=norm_num_groups,
                        cross_attention_dim=cross_attention_dim if use_cross_attn_cond else None,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        is_rollout=is_rollout,
                    ))
                elif isinstance(mod, ResnetBlock2D):
                    _set_module(self, name, TriplaneResBlock(
                        channels=mod.in_channels,
                        emb_channels=mod.time_emb_proj.in_features,
                        out_channels=mod.out_channels,
                        use_scale_shift_norm=mod.time_embedding_norm == "scale_shift",
                        is_rollout=is_rollout,
                    ))
                elif isinstance(mod, Downsample2D):
                    _set_module(self, name, TriplaneDownsample2x(
                        in_channels=mod.channels,
                        out_channels=mod.out_channels,
                        conv_down=conv_down,
                        tri_z_down=tri_z_down,
                    ))
                elif isinstance(mod, Upsample2D):
                    _set_module(self, name, TriplaneUpsample2x(
                        in_channels=mod.channels,
                        out_channels=mod.out_channels,
                        conv_up=conv_up,
                        tri_z_down=tri_z_down,
                        output_padding=0 if 'up_blocks.0' in name else 1,
                    ))
                elif isinstance(mod, nn.Conv2d):
                    _set_module(self, name, TriplaneConv(
                        channels=mod.in_channels, 
                        out_channels=mod.out_channels, 
                        kernel_size=mod.kernel_size,
                        padding=mod.padding,
                        is_rollout=is_rollout,
                    ))
                elif isinstance(mod, nn.GroupNorm):
                    _set_module(self, name, TriplaneNorm(
                        num_groups=norm_num_groups,
                        num_channels=mod.num_channels,
                    ))
                elif isinstance(mod, nn.SiLU):
                    _set_module(self, name, TriplaneSiLU()
                    )
        logging.debug(
            f"[UNet2DConditionModelOcc]: {self}")
        self.trainable_state = "all"

        # triplane_map conditioning embedding
        self.use_map_cond = use_map_cond
        if self.use_map_cond:
            if map_embedder_cls is None:
                cond_embedder_cls = TriplaneMapConditioningEmbedding
                embedder_param = {
                    "conditioning_size": map_size_bev_seg if map_cond_type == 'bev_seg' else map_size_hd_map,
                    "block_out_channels": conditioning_embedding_out_channels,
                    "is_rollout": is_rollout,
                    "downsample": tri_size[0] == 100,
                    "conv_down": conv_down,
                    "tri_z_down": tri_z_down,
                }
            else:
                cond_embedder_cls = load_module(map_embedder_cls)
                embedder_param = map_embedder_param
            self.map_cond_embedding = cond_embedder_cls(
                conditioning_embedding_channels=block_out_channels[0],
                **embedder_param,
            )
            logging.debug(
                f"[UNet2DConditionModelOcc] map_embedder: {self.map_cond_embedding}")

            # uncond_map
            self.drop_cond_ratio = drop_cond_ratio
            if use_uncond_map is not None and drop_cond_ratio > 0:
                map_size = map_size_bev_seg if map_cond_type == 'bev_seg' else map_size_hd_map
                if use_uncond_map == "zero":
                    tmp = torch.zeros(map_size)
                    self.register_buffer("uncond_map", tmp)
                elif use_uncond_map == "negative1":
                    tmp = torch.ones(map_size)
                    self.register_buffer("uncond_map", -tmp)  # -1
                elif use_uncond_map == "random":
                    tmp = torch.randn(map_size)
                    self.register_buffer("uncond_map", tmp)
                elif use_uncond_map == "learnable":
                    tmp = nn.Parameter(torch.randn(map_size))
                    self.register_parameter("uncond_map", tmp)
                else:
                    raise TypeError(f"Unknown map type: {use_uncond_map}.")
            else:
                self.uncond_map = None

        # BEV bbox embedder
        self.use_cross_attn_cond = use_cross_attn_cond
        if self.use_cross_attn_cond:
            model_cls = load_module(bbox_embedder_cls)
            self.bbox_embedder = model_cls(**bbox_embedder_param)

    @property
    def trainable_module(self) -> Dict[str, nn.Module]:
        if self.trainable_state == "all":
            return {self.__class__: self}
        elif self.trainable_state == "only_new":
            return self._new_module
        else:
            raise ValueError(f"Unknown trainable_state: {self.trainable_state}")

    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        for mod in self.trainable_module.values():
            for param in mod.parameters():
                params.append(param)
        return params

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        # first, set all to false
        super().train(False)
        if mode:
            # ensure gradient_checkpointing is usable, set training = True
            for mod in self.modules():
                if getattr(mod, "gradient_checkpointing", False):
                    mod.training = True
        # then, for some modules, we set according to `mode`
        self.training = False
        for mod in self.trainable_module.values():
            if mod is self:
                super().train(mode)
            else:
                mod.train(mode)
        return self

    def enable_gradient_checkpointing(self, flag=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        # self.apply(partial(self._set_gradient_checkpointing, value=True))
        mod_idx = -1
        for module in self.modules():
            if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D)):
                mod_idx += 1
                if flag is not None and not flag[mod_idx]:
                    logging.debug(
                        f"[UNet2DConditionModelOcc] "
                        f"gradient_checkpointing skip [{module.__class__}]")
                    continue
                logging.debug(f"[UNet2DConditionModelOcc] set "
                              f"[{module.__class__}] to gradient_checkpointing")
                module.gradient_checkpointing = True

    def add_uncond_to_kwargs(
            self, bboxes_3d_data: dict, image, max_len=None,
            **kwargs):
        # uncond in the front, cond in the tail
        batch_size = image.shape[:1]
        ret = dict()

        if self.use_cross_attn_cond:
            if bboxes_3d_data is None:
                logging.warn(
                    "Your 'bboxes_3d_data' should not be None. If this warning keeps "
                    "popping, please check your code.")
                if max_len is not None:
                    device = image.device
                    # fmt: off
                    ret["bboxes_3d_data"] = {
                        "bboxes": torch.zeros([batch_size * 2, max_len, 8, 3], device=device),
                        "classes": torch.zeros([batch_size * 2, max_len], device=device, dtype=torch.long),
                        "masks": torch.zeros([batch_size * 2, max_len], device=device, dtype=torch.bool),
                    }
                    # fmt: on
                    for k, v in ret["bboxes_3d_data"].items():
                        logging.debug(f"padding {k} to {v.shape}.")
                else:
                    ret["bboxes_3d_data"] = None
            else:
                ret["bboxes_3d_data"] = dict()  # do not change the original dict
                for key in ["bboxes", "classes", "masks"]:
                    ret["bboxes_3d_data"][key] = torch.cat([
                        torch.zeros_like(bboxes_3d_data[key]),
                        bboxes_3d_data[key],
                    ])
                    if max_len is not None:
                        token_num = max_len - ret["bboxes_3d_data"][key].shape[2]
                        assert token_num >= 0
                        to_pad = torch.zeros_like(ret["bboxes_3d_data"][key])
                        to_pad = repeat(
                            to_pad[:, :, 1], 'b n ... -> b n l ...', l=token_num)
                        ret["bboxes_3d_data"][key] = torch.cat([
                            ret["bboxes_3d_data"][key], to_pad,
                        ], dim=2)
                        logging.debug(
                            f"padding {key} with {token_num}, final size: "
                            f"{ret['bboxes_3d_data'][key].shape}")

        if self.use_map_cond:
            if self.uncond_map is None:
                ret['image'] = image
            else:
                ret['image'] = self.substitute_with_uncond_map(image, None)

        # others, keep original
        for k, v in kwargs.items():
            ret[k] = v
        return ret

    def _random_use_uncond(
            self, encoder_hidden_states, encoder_hidden_states_uncond):
        """
        Args:
            encoder_hidden_states (_type_): B, max_len, C
            encoder_hidden_states_uncond (_type_): 1, max_len, C
        """
        # uncond prompt with camera
        assert self.drop_cond_ratio > 0.0 and self.training
        # mask: 1 -> use uncond, 0 -> keep original
        mask = torch.zeros(
            encoder_hidden_states.shape[:1],
            dtype=torch.long)
        for bs in range(len(encoder_hidden_states)):
            # in each batch, we may randomly select one camera to drop
            if random.random() < self.drop_cond_ratio:
                mask[bs] = 1
        mask = mask > 0  # only bool can index as mask
        encoder_hidden_states[mask] = encoder_hidden_states_uncond[None]
        return encoder_hidden_states, mask

    def substitute_with_uncond_map(self, controlnet_cond, mask=None):
        """_summary_

        Args:
            controlnet_cond (Tensor): map with B, C, H, W
            mask (LongTensor): binary mask on B dim

        Returns:
            Tensor: controlnet_cond
        """
        if mask is None:  # all to uncond
            mask = torch.ones(controlnet_cond.shape[0], dtype=torch.long)
        if any(mask > 0) and self.uncond_map is None:
            raise RuntimeError(f"You cannot use uncond_map before setting it.")
        if all(mask == 0):
            return controlnet_cond
        controlnet_cond[mask > 0] = self.uncond_map[None]
        return controlnet_cond

    def _random_use_uncond_map(self, controlnet_cond, mask=None):
        """randomly replace map to unconditional map (if not None)

        Args:
            controlnet_cond (Tensor): B, C, H=200, W=200

        Returns:
            Tensor: controlnet_cond
        """
        if self.uncond_map is None:
            return controlnet_cond
        if mask is None:
            mask = torch.zeros(len(controlnet_cond), dtype=torch.long)
            for i in range(len(mask)):
                if random.random() < self.drop_cond_ratio:
                    mask[i] = 1
        else:
            mask = mask.type(torch.long)
        return self.substitute_with_uncond_map(controlnet_cond, mask)
    
    def prepare(self, cfg, **kwargs):
        self.bbox_embedder.prepare(cfg, **kwargs)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_uncond: torch.Tensor = None,
        triplane_map_cond: torch.FloatTensor = None,
        bboxes_3d_data: Dict[str, Any] = None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # TODO: actually, we do not change logic in forward

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            if self._WARN_ONCE == 0:
                logging.warning(
                    "[UNet2DConditionModelOcc] Forward upsample size to force interpolation output size.")
                self._WARN_ONCE = 1
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        
        # we may drop the condition during training, but not drop controlnet
        if (self.drop_cond_ratio > 0.0 and self.training):
            if self.use_cross_attn_cond and encoder_hidden_states_uncond is not None:
                encoder_hidden_states, uncond_mask = self._random_use_uncond(
                    encoder_hidden_states, encoder_hidden_states_uncond)
            else:
                uncond_mask = None
            if self.use_map_cond:
                triplane_map_cond = triplane_map_cond.type(self.dtype)
                triplane_map_cond = self._random_use_uncond_map(triplane_map_cond, uncond_mask)
        else:
            uncond_mask = None

        # Scene Edit Inference using cls_id
        # cls_id = 0
        # get cls_id -> bool_mask
        # mask_to_clear = (bboxes_3d_data['classes'] == cls_id)
        # bboxes_3d_data['masks'][mask_to_clear] = False

        # 0.5. bbox embeddings
        # bboxes data should follow the format of (B, N_cam or 1, max_len, ...)
        # for each view
        if self.use_cross_attn_cond and bboxes_3d_data is not None:
            bbox_embedder_kwargs = {}
            for k, v in bboxes_3d_data.items():
                bbox_embedder_kwargs[k] = rearrange(
                    v.clone(), 'b n ... -> (b n) ...')  # view_shared: B,Ncam=1,Nbox,C -> B,Nbox,C
            if uncond_mask is not None:
                masks = bbox_embedder_kwargs["masks"]
                masks[uncond_mask > 0] = 0
            bbox_emb = self.bbox_embedder(**bbox_embedder_kwargs)
            encoder_hidden_states = torch.cat([
                encoder_hidden_states, bbox_emb # B, Ncap+Nbox, C
            ], dim=1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor(
                [timesteps],
                dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.use_cross_attn_cond and self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.use_cross_attn_cond and self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)

        # Scene Edit Inference using cls_id
        # triplane_map_cond[:, cls_id, ...] = 0

        # triplane_map_cond
        if self.use_map_cond and triplane_map_cond is not None:
            triplane_map_cond = self.map_cond_embedding(triplane_map_cond)
            sample += triplane_map_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=copy.deepcopy(cross_attention_kwargs),
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=copy.deepcopy(cross_attention_kwargs),
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(
                upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample, temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=copy.deepcopy(cross_attention_kwargs),
                    upsample_size=upsample_size, attention_mask=attention_mask,)
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size)

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample