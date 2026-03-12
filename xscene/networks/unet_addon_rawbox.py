from typing import Any, Dict, Optional, Tuple, Union, List
import logging

import random
import torch
import torch.nn as nn
import numpy as np
from einops import repeat, rearrange

from diffusers import UNet2DConditionModel
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.models.controlnet import zero_module
from diffusers.models.attention import BasicTransformerBlock

from .embedder import get_embedder
from .output_cls import BEVControlNetOutput
from .map_embedder import BEVControlNetConditioningEmbedding, ControlNetConditioningEmbedding
from ..misc.common import load_module, _get_module, _set_module
from .blocks import (
    TransformerBlockT5,
    Transformer2DModelT5,
    CrossAttnDownBlock2DT5,
    UNetMidBlock2DCrossAttnT5,
)


class BEVControlNetModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = None,
        # these two kwargs will be used in `self.config`
        global_pool_conditions: bool = False,
        # BEV params
        uncond_cam_in_dim: Tuple[int, int] = (3, 7),
        camera_in_dim: int = 189,
        camera_out_dim: int = 768,  # same as word embeddings
        map_embedder_cls: str = None,
        map_embedder_param: dict = None,
        map_size: Tuple[int, int, int] = None,
        use_uncond_map: str = None,
        drop_cond_ratio: float = 0.0,
        drop_cam_num: int = 1,
        drop_cam_with_box: bool = False,
        cam_embedder_param: Optional[Dict] = None,
        bbox_embedder_cls: str = None,
        bbox_embedder_param: dict = None,
        with_layout_canvas: bool = False,
        canvas_conditioning_channels: int = 14,
        canvas_size: Tuple[int, int, int] = (14, 224, 400),
        with_occ_render_img: bool = False,
        occrender_conditioning_channels: int = 20,
        render_img_size: Tuple[int, int, int] = (20, 224, 400),
        render_depth_size: Tuple[int, int, int] = (1, 224, 400),
        occrender_embedding_out_channels: Tuple[int, int, int, int, int] = (16, 32, 64, 96, 256),
        occrender_output_size: Tuple[int, int] = None,
    ):
        super().__init__()
        logging.debug(
            "[BEVControlNetModel] instantiating your own version of controlnet."
        )

        # Check inputs
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(
            only_cross_attention
        ) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(
                attention_head_dim, int) and len(attention_head_dim) != len(
                down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        # BEV camera
        self.cam2token = nn.Linear(camera_in_dim, camera_out_dim)
        # TODO: how to initilize this?
        if uncond_cam_in_dim:
            self.uncond_cam = nn.Embedding(
                1, uncond_cam_in_dim[0] * uncond_cam_in_dim[1])
            # the num of len-3 vectors. We use Fourier emb on len-3 vector.
            self.uncond_cam_num = uncond_cam_in_dim[1]
        self.drop_cond_ratio = drop_cond_ratio
        self.drop_cam_num = drop_cam_num
        self.drop_cam_with_box = drop_cam_with_box

        # in 3, freq 4 -> embedder.out_dim = 27
        self.cam_embedder = get_embedder(**cam_embedder_param)

        # input
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=conv_in_kernel,
            padding=conv_in_padding,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(
            block_out_channels[0],
            flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(
                num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(
                timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(
                projection_class_embeddings_input_dim, time_embed_dim
            )
        else:
            self.class_embedding = None

        # control net conditioning embedding
        if map_embedder_cls is None:
            cond_embedder_cls = BEVControlNetConditioningEmbedding
            embedder_param = {
                "conditioning_size": map_size,
                "block_out_channels": conditioning_embedding_out_channels,
            }
        else:
            cond_embedder_cls = load_module(map_embedder_cls)
            embedder_param = map_embedder_param
        self.controlnet_cond_embedding = cond_embedder_cls(
            conditioning_embedding_channels=block_out_channels[0],
            **embedder_param,
        )
        logging.debug(
            f"[BEVControlNetModel] map_embedder: {self.controlnet_cond_embedding}")

        if with_layout_canvas:
            self.layout_canvas_embedding = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=block_out_channels[0],
                block_out_channels=conditioning_embedding_out_channels,
                conditioning_channels=canvas_conditioning_channels,
            )
            logging.debug(
                f"[BEVControlNetModel] canvas_embedder: {self.layout_canvas_embedding}")

        if with_occ_render_img:
            self.occrender_img_embedding = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=block_out_channels[0],
                block_out_channels=occrender_embedding_out_channels,
                conditioning_channels=occrender_conditioning_channels,
                output_size=occrender_output_size,
            )
            logging.debug(
                f"[BEVControlNetModel] occ_render_img_embedder: {self.occrender_img_embedding}")
            
            self.occrender_depth_embedding = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=block_out_channels[0],
                block_out_channels=occrender_embedding_out_channels,
                conditioning_channels=1,
                output_size=occrender_output_size,
            )
            logging.debug(
                f"[BEVControlNetModel] occ_render_depth_embedder: {self.occrender_depth_embedding}")

        # uncond_map
        self.drop_cond_ratio = drop_cond_ratio
        if use_uncond_map is not None and drop_cond_ratio > 0:
            if use_uncond_map == "zero":
                tmp = torch.zeros(map_size)
                self.register_buffer("uncond_map", tmp)
                tmp = torch.zeros(canvas_size)
                self.register_buffer("uncond_canvas", tmp)
                tmp = torch.zeros(render_img_size)
                self.register_buffer("uncond_render_img", tmp)
                tmp = torch.zeros(render_depth_size)
                self.register_buffer("uncond_render_depth", tmp)
            else:
                raise TypeError(f"Unknown map type: {use_uncond_map}.")
        else:
            self.uncond_map = None
            self.uncond_canvas = None
            self.uncond_render_img = None
            self.uncond_render_depth = None
        self.uncond_dict = {
            "bevmap": self.uncond_map,
            "canvas": self.uncond_canvas,
            "render_img": self.uncond_render_img,
            "render_depth": self.uncond_render_depth,
        }

        # BEV bbox embedder
        model_cls = load_module(bbox_embedder_cls)
        self.bbox_embedder = model_cls(**bbox_embedder_param)

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [
                only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]

        controlnet_block = nn.Conv2d(
            output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2d(
                    output_channel, output_channel, kernel_size=1
                )
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv2d(
                    output_channel, output_channel, kernel_size=1
                )
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv2d(
            mid_block_channel, mid_block_channel, kernel_size=1
        )
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

        # change the original diffusion module to new module to support T5 cross-attn and multiview cross-attn
        for name, mod in list(self.named_modules()):
            if isinstance(mod, CrossAttnDownBlock2D):
                _set_module(self, name, CrossAttnDownBlock2DT5(**mod._args,))
            elif isinstance(mod, UNetMidBlock2DCrossAttn):
                _set_module(self, name, UNetMidBlock2DCrossAttnT5(**mod._args,))
            elif isinstance(mod, Transformer2DModel):
                _set_module(self, name, Transformer2DModelT5(**mod._args,))
        # set the new module to be trainable: T5 cross-attn and multiview cross-attn
        for name, mod in list(self.named_modules()):
            if isinstance(mod, BasicTransformerBlock):
                _set_module(self, name, TransformerBlockT5(
                    **mod._args,
                ))

    def _embed_camera(self, camera_param):
        """
        Args:
            camera_param (torch.Tensor): [N, 6, 3, 7], 7 for 3 + 4
        """
        (bs, N_cam, C_param, emb_num) = camera_param.shape
        assert C_param == 3
        assert emb_num == self.uncond_cam_num or self.uncond_cam_num is None, (
            f"You assign `uncond_cam_in_dim[1]={self.uncond_cam_num}`, "
            f"but your data actually have {emb_num} to embed. Please change your config."
        )
        camera_emb = self.cam_embedder(
            rearrange(camera_param, "b n d c -> (b n c) d")
        )
        camera_emb = rearrange(
            camera_emb, "(b n c) d -> b n (c d)", n=N_cam, b=bs
        )
        return camera_emb

    def uncond_cam_param(self, repeat_size: Union[List[int], int] = 1):
        if isinstance(repeat_size, int):
            repeat_size = [1, repeat_size]
        repeat_size_sum = int(np.prod(repeat_size))
        # we only have one uncond cam, embedding input is always 0
        param = self.uncond_cam(torch.LongTensor(
            [0] * repeat_size_sum).to(device=self.device))
        param = param.reshape(*repeat_size, -1, self.uncond_cam_num)
        return param

    def add_cam_states(self, encoder_hidden_states, camera_emb=None):
        """
        Args:
            encoder_hidden_states (torch.Tensor): b, len, 768
            camera_emb (torch.Tensor): b, n_cam, dim. if None, use uncond cam.
        """
        bs = encoder_hidden_states.shape[0]
        if camera_emb is None:
            # B, 1, 768
            cam_hidden_states = self.cam2token(self._embed_camera(
                self.uncond_cam_param(bs)))
        else:
            cam_hidden_states = self.cam2token(camera_emb)  # B, N_cam, dim
        N_cam = cam_hidden_states.shape[1]
        encoder_hidden_states_with_cam = torch.cat([
            cam_hidden_states.unsqueeze(2),  # B, N_cam, 1, 768
            repeat(encoder_hidden_states, 'b c ... -> b repeat c ...',
                   repeat=N_cam)
        ], dim=2)  # B, N, len + 1, dim
        return encoder_hidden_states_with_cam

    def substitute_with_uncond_cam(
        self,
        encoder_hidden_states_with_cam,
        encoder_hidden_states_uncond,
        mask: Optional[torch.LongTensor] = None,
    ):
        encoder_hidden_states_uncond_with_cam = self.add_cam_states(
            encoder_hidden_states_uncond)
        if mask is None:  # all to uncond
            mask = torch.ones(
                encoder_hidden_states_with_cam.shape[: 2],
                dtype=torch.long)
        mask = mask > 0  # only bool can index as mask
        encoder_hidden_states_with_cam[mask] = encoder_hidden_states_uncond_with_cam[None]
        return encoder_hidden_states_with_cam

    def _random_use_uncond_cam(
            self, encoder_hidden_states_with_cam, encoder_hidden_states_uncond, mask=None):
        """
        Args:
            encoder_hidden_states_with_cam (_type_): B, N, max_len + 1, 768
            encoder_hidden_states_uncond (_type_): 1, max_len, 768
        """
        # uncond prompt with camera
        assert self.drop_cond_ratio > 0.0 and self.training
        # mask: 1 -> use uncond, 0 -> keep original
        if mask is None:
            mask = torch.zeros(
                encoder_hidden_states_with_cam.shape[: 2],
                dtype=torch.long)
            for bs in range(len(encoder_hidden_states_with_cam)):
                # in each batch, we may randomly select one camera to drop
                if random.random() < self.drop_cond_ratio:
                    cam_id = random.sample(
                        range(encoder_hidden_states_with_cam.shape[1]),
                        self.drop_cam_num)
                    mask[bs, cam_id] = 1
        encoder_hidden_states_with_cam = self.substitute_with_uncond_cam(
            encoder_hidden_states_with_cam, encoder_hidden_states_uncond, mask)
        return encoder_hidden_states_with_cam, mask

    def substitute_with_uncond_map(self, controlnet_cond, mask=None, cond_type="bevmap"):
        """_summary_

        Args:
            controlnet_cond (Tensor): map with B, C, H, W
            mask (LongTensor): binary mask on B dim

        Returns:
            Tensor: controlnet_cond
        """
        if mask is None:  # all to uncond
            mask = torch.ones(controlnet_cond.shape[0], dtype=torch.long)
        if any(mask > 0) and self.uncond_dict[cond_type] is None:
            raise RuntimeError(f"You cannot use uncond_map before setting it.")
        if all(mask == 0):
            return controlnet_cond
        controlnet_cond[mask > 0] = self.uncond_dict[cond_type][None]
        return controlnet_cond

    def _random_use_uncond_map(self, controlnet_cond, mask=None, cond_type="bevmap"):
        """randomly replace map to unconditional map (if not None)

        Args:
            controlnet_cond (Tensor): B, C, H=200, W=200

        Returns:
            Tensor: controlnet_cond
        """
        if self.uncond_dict[cond_type] is None:
            return controlnet_cond
        if mask is None:
            mask = torch.zeros(len(controlnet_cond), dtype=torch.long)
            for i in range(len(mask)):
                if random.random() < self.drop_cond_ratio:
                    mask[i] = 1
        else:
            mask = mask.type(torch.long)
        return self.substitute_with_uncond_map(controlnet_cond, mask, cond_type)

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",
        load_weights_from_unet: bool = True,
        load_param: bool = False,
        **kwargs
    ):
        r"""
        Instantiate BEVControlnet class from UNet2DConditionModel.

        Parameters:
            unet (`UNet2DConditionModel`):
                UNet model which weights are copied to the ControlNet. Note that all configuration options are also
                copied where applicable.
        """

        bev_controlnet = cls(
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            # BEV
            **kwargs,
        )

        if load_weights_from_unet and not load_param:
            bev_controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            bev_controlnet.time_proj.load_state_dict(
                unet.time_proj.state_dict())
            bev_controlnet.time_embedding.load_state_dict(
                unet.time_embedding.state_dict()
            )

            if bev_controlnet.class_embedding:
                bev_controlnet.class_embedding.load_state_dict(
                    unet.class_embedding.state_dict()
                )

            bev_controlnet.down_blocks.load_state_dict(
                unet.down_blocks.state_dict())
            bev_controlnet.mid_block.load_state_dict(
                unet.mid_block.state_dict())

        if load_param:
            s = bev_controlnet.state_dict()
            for key, val in unet.state_dict().items():

                # process ckpt from parallel module
                if key[:6] == 'module':
                    key = key[7:]

                if key in s and s[key].shape == val.shape:
                    s[key][...] = val
                elif key not in s:
                    print('ignore weight from not found key {}'.format(key))
                else:
                    print('ignore weight of mismached shape in key {}'.format(key))

            bev_controlnet.load_state_dict(s)

        return bev_controlnet

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(
                    f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
            self,
            processor: Union
            [AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Parameters:
            `processor (`dict` of `AttentionProcessor` or `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `Attention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainable attention processors.:

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes.")

        def fn_recursive_attn_processor(
                name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(
                    f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = (
            num_sliceable_layers * [slice_size]
            if not isinstance(slice_size, list)
            else slice_size
        )

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}.")

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(
                    f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(
            module: torch.nn.Module, slice_size: List[int]
        ):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value

    def add_uncond_to_kwargs(
            self, camera_param, bboxes_3d_data: dict, image, max_len=None,
            **kwargs):
        # uncond in the front, cond in the tail
        batch_size, N_cam = camera_param.shape[:2]
        ret = dict()

        ret['camera_param'] = torch.cat([
            self.uncond_cam_param([batch_size, N_cam]),
            camera_param,
        ])

        if bboxes_3d_data is None:
            logging.warn(
                "Your 'bboxes_3d_data' should not be None. If this warning keeps "
                "popping, please check your code.")
            if max_len is not None:
                device = camera_param.device
                # fmt: off
                ret["bboxes_3d_data"] = {
                    "bboxes": torch.zeros([batch_size * 2, N_cam, max_len, 8, 3], device=device),
                    "classes": torch.zeros([batch_size * 2, N_cam, max_len], device=device, dtype=torch.long),
                    "masks": torch.zeros([batch_size * 2, N_cam, max_len], device=device, dtype=torch.bool),
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

        if self.uncond_map is None:
            ret['image'] = image
        else:
            ret['image'] = self.substitute_with_uncond_map(image, None)

        # others, keep original
        for k, v in kwargs.items():
            ret[k] = v
        return ret

    def add_uncond_to_emb(
        self, prompt_embeds, N_cam, encoder_hidden_states_with_cam
    ):
        # uncond in the front, cond in the tail
        encoder_hidden_states_with_uncond_cam = self.controlnet.add_cam_states(
            prompt_embeds)
        token_num = encoder_hidden_states_with_cam.shape[1] - \
            encoder_hidden_states_with_uncond_cam.shape[1]
        encoder_hidden_states_with_uncond_cam = self.bbox_embedder.add_n_uncond_tokens(
            encoder_hidden_states_with_uncond_cam, token_num)

        encoder_hidden_states_with_cam = torch.cat([
            repeat(
                encoder_hidden_states_with_uncond_cam,
                'b ... -> (b n) ...', n=N_cam,
            ),
            encoder_hidden_states_with_cam,
        ], dim=0)
        return encoder_hidden_states_with_cam

    def prepare(self, cfg, **kwargs):
        self.bbox_embedder.prepare(cfg, **kwargs)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        camera_param: torch.Tensor,  # BEV
        bboxes_3d_data: Dict[str, Any],  # BEV
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_t5: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        encoder_hidden_states_uncond: torch.Tensor = None,  # BEV
        encoder_hidden_states_uncond_t5: torch.Tensor = None,
        scene_embedding: Optional[torch.Tensor] = None,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[BEVControlNetOutput, Tuple]:
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        assert isinstance(controlnet_cond, list)
        if len(controlnet_cond) == 5:
            _, controlnet_cond, layout_canvas, occ_render_img, occ_render_depth = controlnet_cond[:5]
        elif len(controlnet_cond) == 3:
            _, controlnet_cond, layout_canvas = controlnet_cond[:3]
            occ_render_img, occ_render_depth = None, None
        elif len(controlnet_cond) == 1:
            controlnet_cond = controlnet_cond[:1]
            layout_canvas, occ_render_img, occ_render_depth = None, None, None

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(
                f"unknown `controlnet_conditioning_channel_order`: {channel_order}"
            )

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. camera
        N_cam = camera_param.shape[1]
        camera_emb = self._embed_camera(camera_param)
        # (B, N_cam, max_len + 1, dim=768)
        encoder_hidden_states_with_cam = self.add_cam_states(
            encoder_hidden_states, camera_emb
        )
        encoder_hidden_states_t5_with_cam = self.add_cam_states(
            encoder_hidden_states_t5, camera_emb
        )
        # we may drop the condition during training, but not drop controlnet
        if (self.drop_cond_ratio > 0.0 and self.training):
            if encoder_hidden_states_uncond is not None:
                encoder_hidden_states_with_cam, uncond_mask = self._random_use_uncond_cam(
                    encoder_hidden_states_with_cam, encoder_hidden_states_uncond)
                encoder_hidden_states_t5_with_cam, _ = self._random_use_uncond_cam(
                    encoder_hidden_states_t5_with_cam, encoder_hidden_states_uncond_t5, uncond_mask)
            controlnet_cond = controlnet_cond.type(self.dtype)
            controlnet_cond = self._random_use_uncond_map(controlnet_cond)
        else:
            uncond_mask = None

        # # Scene Edit Inference use clas_id
        # cls_id = 0
        # mask_to_clear = (bboxes_3d_data['classes'] == cls_id)
        # bboxes_3d_data['masks'][mask_to_clear] = False

        # 0.5. bbox embeddings
        # bboxes data should follow the format of (B, N_cam or 1, max_len, ...)
        # for each view
        if bboxes_3d_data is not None:
            bbox_embedder_kwargs = {}
            for k, v in bboxes_3d_data.items():
                bbox_embedder_kwargs[k] = v.clone()
            if self.drop_cam_with_box and uncond_mask is not None:
                _, n_box = bboxes_3d_data["bboxes"].shape[:2]
                if n_box != N_cam:
                    assert n_box == 1, "either N_cam or 1."
                    for k in bboxes_3d_data.keys():
                        ori_v = rearrange(
                            bbox_embedder_kwargs[k], 'b n ... -> (b n) ...')
                        new_v = repeat(ori_v, 'b ... -> b n ...', n=N_cam)
                        bbox_embedder_kwargs[k] = new_v
                # here we set mask for dropped boxes to all zero
                masks = bbox_embedder_kwargs['masks']
                masks[uncond_mask > 0] = 0
            # original flow
            b_box, n_box = bbox_embedder_kwargs["bboxes"].shape[:2]
            for k in bboxes_3d_data.keys():
                bbox_embedder_kwargs[k] = rearrange(
                    bbox_embedder_kwargs[k], 'b n ... -> (b n) ...')
            bbox_emb = self.bbox_embedder(**bbox_embedder_kwargs)
            if n_box != N_cam:
                # n_box should be 1: all views share the same set of bboxes, we repeat
                bbox_emb = repeat(bbox_emb, 'b ... -> b n ...', n=N_cam)
            else:
                # each view already has its set of bboxes
                bbox_emb = rearrange(bbox_emb, '(b n) ... -> b n ...', n=N_cam)
            encoder_hidden_states_with_cam = torch.cat([
                encoder_hidden_states_with_cam, bbox_emb
            ], dim=2)

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
        # timesteps = timesteps.expand(sample.shape[0])

        timesteps = timesteps.reshape(-1)  # time_proj can only take 1-D input
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # BEV: we remap data to have (B n) as batch size
        sample = rearrange(sample, 'b n ... -> (b n) ...')
        encoder_hidden_states_with_cam = rearrange(
            encoder_hidden_states_with_cam, 'b n ... -> (b n) ...')
        encoder_hidden_states_t5_with_cam = rearrange(
            encoder_hidden_states_t5_with_cam, 'b n ... -> (b n) ...')
        if len(emb) < len(sample):
            emb = repeat(emb, 'b ... -> (b repeat) ...', repeat=N_cam)
        controlnet_cond = repeat(
            controlnet_cond, 'b ... -> (b repeat) ...', repeat=N_cam)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.1 bev_hdmap condition
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample += controlnet_cond
        # 2.2 layout+box perspective canvas condition
        if layout_canvas is not None:
            # Scene Edit Inference use cls_id
            # layout_canvas[:,:, cls_id+4, ...] = 0
            sample += self.layout_canvas_embedding(rearrange(layout_canvas, 'b n ... -> (b n) ...'))
        # 2.3 scene positional embedding condition
        if scene_embedding is not None:
            sample += rearrange(scene_embedding, 'b n ... -> (b n) ...')
        # 2.4 occ render image condition
        if occ_render_img is not None:
            # Scene Edit Inference use cls_id
            # occ_render_img[:,:, cls_id+1, ...] = 0
            sample += self.occrender_img_embedding(rearrange(occ_render_img, 'b n ... -> (b n) ...'))
        # 2.5 occ render depth condition
        if occ_render_depth is not None:
            sample += self.occrender_depth_embedding(rearrange(occ_render_depth, 'b n ... -> (b n) ...'))


        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_with_cam,
                    encoder_hidden_states_t5=encoder_hidden_states_t5_with_cam,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states_with_cam,
                encoder_hidden_states_t5=encoder_hidden_states_t5_with_cam,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(
            down_block_res_samples, self.controlnet_down_blocks
        ):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples += (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode:
            scales = torch.logspace(
                -1, 0, len(down_block_res_samples) + 1
            )  # 0.1 to 1.0
            scales *= conditioning_scale
            down_block_res_samples = [
                sample * scale for sample,
                scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample *= scales[-1]  # last one
        else:
            down_block_res_samples = [
                sample * conditioning_scale for sample in down_block_res_samples
            ]
            mid_block_res_sample *= conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True)
                for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(
                mid_block_res_sample, dim=(2, 3), keepdim=True
            )

        if not return_dict:
            return (
                down_block_res_samples,
                mid_block_res_sample,
                encoder_hidden_states_with_cam,
                encoder_hidden_states_t5_with_cam,
            )

        return BEVControlNetOutput(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
            encoder_hidden_states_with_cam=encoder_hidden_states_with_cam,
            encoder_hidden_states_t5_with_cam=encoder_hidden_states_t5_with_cam,
        )
