from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
import math

from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    XFormersAttnProcessor,
    XFormersAttnAddedKVProcessor,
    CustomDiffusionXFormersAttnProcessor,
    LoRAXFormersAttnProcessor,
)
from diffusers.models.attention import BasicTransformerBlock, AdaLayerNorm
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn
from diffusers.models.transformer_2d import Transformer2DModel, Transformer2DModelOutput
from diffusers.models.controlnet import zero_module
from diffusers.utils import is_torch_version
from xscene.networks.utils import normalization, checkpoint, linear, compose_featmaps, decompose_featmaps, SiLU
from ..misc.common import _get_module

def is_xformers(module):
    return isinstance(module.processor, (
        XFormersAttnProcessor,
        XFormersAttnAddedKVProcessor,
        CustomDiffusionXFormersAttnProcessor,
        LoRAXFormersAttnProcessor,
    ))

def _ensure_kv_is_int(view_pair: dict):
    """yaml key can be int, while json cannot. We convert here.
    """
    new_dict = {}
    for k, v in view_pair.items():
        new_value = [int(vi) for vi in v]
        new_dict[int(k)] = new_value
    return new_dict

def get_zero_module(zero_module_type, dim):
    if zero_module_type == "zero_linear":
        # NOTE: zero_module cannot apply to successive layers.
        connector = zero_module(nn.Linear(dim, dim))
    elif zero_module_type == "gated":
        connector = GatedConnector(dim)
    elif zero_module_type == "none":
        # TODO: if this block is in controlnet, we may not need zero here.
        def connector(x): return x
    else:
        raise TypeError(f"Unknown zero module type: {zero_module_type}")
    return connector
class TriplaneConv(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, padding, is_rollout=True) -> None:
        super().__init__()
        in_channels = channels * 3 if is_rollout else channels
        self.is_rollout = is_rollout

        self.conv_xy = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_xz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_yz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, featmaps):
        featmaps = decompose_featmaps(featmaps)
        # featmaps: [B,C,X,Y] [B,C,X,Z] [B,C,Y,Z]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        if self.is_rollout:
            tpl_xy_h = torch.cat([tpl_xy,
                            torch.mean(tpl_yz, dim=-1, keepdim=True).transpose(-1, -2).expand_as(tpl_xy),
                            torch.mean(tpl_xz, dim=-1, keepdim=True).expand_as(tpl_xy)], dim=1) # [B, C * 3, H, W]
            tpl_xz_h = torch.cat([tpl_xz,
                                torch.mean(tpl_xy, dim=-1, keepdim=True).expand_as(tpl_xz),
                                torch.mean(tpl_yz, dim=-2, keepdim=True).expand_as(tpl_xz)], dim=1) # [B, C * 3, H, D]
            tpl_yz_h = torch.cat([tpl_yz,
                            torch.mean(tpl_xy, dim=-2, keepdim=True).transpose(-1, -2).expand_as(tpl_yz),
                            torch.mean(tpl_xz, dim=-2, keepdim=True).expand_as(tpl_yz)], dim=1) # [B, C * 3, W, D]
        else:
            tpl_xy_h = tpl_xy
            tpl_xz_h = tpl_xz
            tpl_yz_h = tpl_yz
        
        assert tpl_xy_h.shape[-2] == H and tpl_xy_h.shape[-1] == W
        assert tpl_xz_h.shape[-2] == H and tpl_xz_h.shape[-1] == D
        assert tpl_yz_h.shape[-2] == W and tpl_yz_h.shape[-1] == D

        if tpl_xy_h.dtype != [param.dtype for param in self.conv_xy.parameters()][0]:
            if tpl_xy_h.dtype == torch.float16:
                tpl_xy_h = self.conv_xy(tpl_xy_h.float())
                tpl_xz_h = self.conv_xz(tpl_xz_h.float())
                tpl_yz_h = self.conv_yz(tpl_yz_h.float())
            else:
                tpl_xy_h = self.conv_xy(tpl_xy_h.half())
                tpl_xz_h = self.conv_xz(tpl_xz_h.half())
                tpl_yz_h = self.conv_yz(tpl_yz_h.half())
        else:
            tpl_xy_h = self.conv_xy(tpl_xy_h)
            tpl_xz_h = self.conv_xz(tpl_xz_h)
            tpl_yz_h = self.conv_yz(tpl_yz_h)

        return compose_featmaps(tpl_xy_h, tpl_xz_h, tpl_yz_h)


class TriplaneNorm(nn.Module):
    def __init__(self,  num_groups, num_channels) -> None:
        super().__init__()
        self.norm_xy = normalization(num_groups, num_channels)
        self.norm_xz = normalization(num_groups, num_channels)
        self.norm_yz = normalization(num_groups, num_channels)

    def forward(self, featmaps):
        featmaps = decompose_featmaps(featmaps)
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        tpl_xy_h = self.norm_xy(tpl_xy) # [B, C, H, W]
        tpl_xz_h = self.norm_xz(tpl_xz) # [B, C, H, D]
        tpl_yz_h = self.norm_yz(tpl_yz) # [B, C, W, D]

        assert tpl_xy_h.shape[-2] == H and tpl_xy_h.shape[-1] == W
        assert tpl_xz_h.shape[-2] == H and tpl_xz_h.shape[-1] == D
        assert tpl_yz_h.shape[-2] == W and tpl_yz_h.shape[-1] == D

        return compose_featmaps(tpl_xy_h, tpl_xz_h, tpl_yz_h)


class TriplaneSiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.silu = SiLU()

    def forward(self, featmaps):
        featmaps = decompose_featmaps(featmaps)
        tpl_xy, tpl_xz, tpl_yz = featmaps
        return compose_featmaps(self.silu(tpl_xy), self.silu(tpl_xz), self.silu(tpl_yz))


class TriplaneDownsample2x(nn.Module):
    def __init__(self, in_channels, out_channels, conv_down, tri_z_down) -> None:
        super().__init__()
        self.conv_down = conv_down
        self.tri_z_down = tri_z_down

        if conv_down :
            if self.tri_z_down:
                self.conv_xy = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, padding_mode='replicate')
                self.conv_xz = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, padding_mode='replicate')
                self.conv_yz = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, padding_mode='replicate')
            else : 
                self.conv_xy = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, padding_mode='replicate')
                self.conv_xz = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=(2, 1), padding_mode='replicate')
                self.conv_yz = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=(2, 1), padding_mode='replicate')
                
    def forward(self, featmaps):
        featmaps = decompose_featmaps(featmaps)
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]
        if self.conv_down:
            tpl_xy = self.conv_xy(tpl_xy)
            tpl_xz = self.conv_xz(tpl_xz)
            tpl_yz = self.conv_yz(tpl_yz)
        else : 
            tpl_xy = F.avg_pool2d(tpl_xy, kernel_size=2, stride=2)
            if self.tri_z_down:
                tpl_xz = F.avg_pool2d(tpl_xz, kernel_size=2, stride=2)
                tpl_yz = F.avg_pool2d(tpl_yz, kernel_size=2, stride=2)
            else : 
                tpl_xz = F.avg_pool2d(tpl_xz, kernel_size=(2, 1), stride=(2, 1))
                tpl_yz = F.avg_pool2d(tpl_yz, kernel_size=(2, 1), stride=(2, 1))
        return compose_featmaps(tpl_xy, tpl_xz, tpl_yz)


class TriplaneUpsample2x(nn.Module):
    def __init__(self, in_channels, out_channels, output_padding=1, conv_up=True, tri_z_down=False) -> None:
        super().__init__()
        self.conv_up = conv_up
        self.tri_z_down = tri_z_down
        
        if conv_up :
            if self.tri_z_down:
                self.conv_xy = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, output_padding=output_padding, stride=2)
                self.conv_xz = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, output_padding=output_padding, stride=2)
                self.conv_yz = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, output_padding=output_padding, stride=2)
            else :
                self.conv_xy = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, output_padding=output_padding, stride=2)
                self.conv_xz = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, output_padding=(output_padding,0), stride=(2, 1))
                self.conv_yz = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, output_padding=(output_padding,0), stride=(2, 1))

    def forward(self, featmaps, output_size=None):
        featmaps = decompose_featmaps(featmaps)
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]
        if self.conv_up:
            tpl_xy = self.conv_xy(tpl_xy)
            tpl_xz = self.conv_xz(tpl_xz)
            tpl_yz = self.conv_yz(tpl_yz)
        else : 
            tpl_xy = F.interpolate(tpl_xy, scale_factor=2, mode='bilinear', align_corners=False)
            if self.tri_z_down:
                tpl_xz = F.interpolate(tpl_xz, scale_factor=2, mode='bilinear', align_corners=False)
                tpl_yz = F.interpolate(tpl_yz, scale_factor=2, mode='bilinear', align_corners=False)
            else :    
                tpl_xz = F.interpolate(tpl_xz, scale_factor=(2, 1), mode='bilinear', align_corners=False)
                tpl_yz = F.interpolate(tpl_yz, scale_factor=(2, 1), mode='bilinear', align_corners=False)
                
        return compose_featmaps(tpl_xy, tpl_xz, tpl_yz)


class TriplaneResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        groups=32,
        use_conv=False,
        use_scale_shift_norm=True,
        use_checkpoint=False,
        up=False,
        down=False,
        is_rollout=True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.groups = groups
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.in_layers = nn.Sequential(
            TriplaneNorm(groups, channels),
            TriplaneSiLU(),
            TriplaneConv(channels, self.out_channels, 3, padding=1, is_rollout=is_rollout),
        )

        self.updown = up or down

        if up:
            self.h_upd = TriplaneUpsample2x(self.out_channels, self.out_channels)
            self.x_upd = TriplaneUpsample2x(self.out_channels, self.out_channels)
        elif down:
            self.h_upd = TriplaneDownsample2x(self.out_channels, self.out_channels)
            self.x_upd = TriplaneDownsample2x(self.out_channels, self.out_channels)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            TriplaneNorm(groups, self.out_channels),
            TriplaneSiLU(),
            # nn.Dropout(p=dropout),
            zero_module(
                TriplaneConv(self.out_channels, self.out_channels, 3, padding=1, is_rollout=is_rollout)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = TriplaneConv(
                channels, self.out_channels, 3, padding=1, is_rollout=False
            )
        else:
            self.skip_connection = TriplaneConv(channels, self.out_channels, 1, padding=0, is_rollout=False)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # x: (h_xy, h_xz, h_yz)
        h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_silu, out_conv = self.out_layers[0], self.out_layers[1], self.out_layers[2]
            scale, shift = torch.chunk(emb_out, 2, dim=1)

            h = out_norm(h)
            h = decompose_featmaps(h)
            h_xy, h_xz, h_yz = h
            h_xy = h_xy * (1 + scale) + shift
            h_xz = h_xz * (1 + scale) + shift
            h_yz = h_yz * (1 + scale) + shift
            h = compose_featmaps(h_xy, h_xz, h_yz)
            # h = out_norm(h) * (1 + scale) + shift

            h = out_silu(h)
            h = out_conv(h)
        else:
            h = decompose_featmaps(h)
            h_xy, h_xz, h_yz = h
            h_xy = h_xy + emb_out
            h_xz = h_xz + emb_out
            h_yz = h_yz + emb_out
            h = compose_featmaps(h_xy, h_xz, h_yz)
            # h = h + emb_out

            h = self.out_layers(h)
        
        x_skip = self.skip_connection(x)
        x_skip_xy, x_skip_xz, x_skip_yz = decompose_featmaps(x_skip)
        h_xy, h_xz, h_yz = decompose_featmaps(h)
        return compose_featmaps(h_xy + x_skip_xy, h_xz + x_skip_xz, h_yz + x_skip_yz)


class TriplaneTransformerModel(nn.Module):
    """
    Transformer model for triplane data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 8): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 16): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """
    def __init__(
        self,
        num_attention_heads: int = 8,   # num_attention_heads
        attention_head_dim: int = 16,   # attention_head_dim
        in_channels: Optional[int] = None,  # in_channels
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,      # norm.num_groups
        cross_attention_dim: Optional[int] = None,  # cross_attention_dim
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        only_cross_attention: bool = False, # only_cross_attention
        upcast_attention: bool = False, # upcast_attention
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        is_rollout: bool = True,    # is_rollout
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Define input layers
        self.in_channels = in_channels
        self.norm = TriplaneNorm(norm_num_groups, in_channels)
        self.proj_in = TriplaneConv(in_channels, inner_dim, kernel_size=1, padding=0, is_rollout=is_rollout)

        # 2. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for d in range(num_layers)
            ]
        )

        # 3. Define output layers
        self.proj_out = TriplaneConv(inner_dim, in_channels, kernel_size=1, padding=0, is_rollout=is_rollout)


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        """
        Args:
            hidden_states ( `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. Input
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)

        # 2. rearrange
        tpl_xy, tpl_xz, tpl_yz = decompose_featmaps(hidden_states)
        B, C, H, W = tpl_xy.shape
        D = tpl_xz.shape[-1]
        tpl_xy = tpl_xy.permute(0, 2, 3, 1).reshape(B, H*W, C)
        tpl_xz = tpl_xz.permute(0, 2, 3, 1).reshape(B, H*D, C)
        tpl_yz = tpl_yz.permute(0, 2, 3, 1).reshape(B, W*D, C)
        hidden_states = torch.concat([tpl_xy, tpl_xz, tpl_yz], dim=1)   # B, H*W + H*D + W*D, C

        # 3. Transoformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 4. Output
        tpl_xy, tpl_xz, tpl_yz = hidden_states[:, :H*W, :], hidden_states[:, H*W:-W*D, :], hidden_states[:, -W*D:, :]
        tpl_xy = tpl_xy.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        tpl_xz = tpl_xz.reshape(B, H, D, C).permute(0, 3, 1, 2).contiguous()
        tpl_yz = tpl_yz.reshape(B, W, D, C).permute(0, 3, 1, 2).contiguous()
        hidden_states = compose_featmaps(tpl_xy, tpl_xz, tpl_yz)
        hidden_states = self.proj_out(hidden_states)

        output = hidden_states + residual
        return (output,)


class GatedConnector(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        data = torch.zeros(dim)
        self.alpha = nn.parameter.Parameter(data)

    def forward(self, inx):
        # as long as last dim of input == dim, pytorch can auto-broad
        return F.tanh(self.alpha) * inx


class BasicMultiviewTransformerBlock(BasicTransformerBlock):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        # multi_view
        neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
        neighboring_attn_type: Optional[str] = "add",
        zero_module_type="zero_linear",
        attn1_q_trainable=False,
    ):
        super().__init__(
            dim, num_attention_heads, attention_head_dim, dropout,
            cross_attention_dim, activation_fn, num_embeds_ada_norm,
            attention_bias, only_cross_attention, double_self_attention,
            upcast_attention, norm_elementwise_affine, norm_type, final_dropout)

        self.neighboring_view_pair = _ensure_kv_is_int(neighboring_view_pair)
        self.neighboring_attn_type = neighboring_attn_type
        # multiview attention
        self.norm4 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        )
        self.attn4 = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        if zero_module_type == "zero_linear":
            # NOTE: zero_module cannot apply to successive layers.
            self.connector = zero_module(nn.Linear(dim, dim))
        elif zero_module_type == "gated":
            self.connector = GatedConnector(dim)
        elif zero_module_type == "none":
            # TODO: if this block is in controlnet, we may not need zero here.
            self.connector = lambda x: x
        else:
            raise TypeError(f"Unknown zero module type: {zero_module_type}")
        self.attn1_q_trainable = attn1_q_trainable

    @property
    def new_module(self):
        ret = {
            "norm4": self.norm4,
            "attn4": self.attn4,
        }
        if isinstance(self.connector, nn.Module):
            ret["connector"] = self.connector
        if self.attn1_q_trainable:
            ret['attn1.to_q'] = self.attn1.to_q
        return ret

    @property
    def n_cam(self):
        return len(self.neighboring_view_pair)

    def _construct_attn_input(self, norm_hidden_states):
        B = len(norm_hidden_states)
        # reshape, key for origin view, value for ref view
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        if self.neighboring_attn_type == "add":
            for key, values in self.neighboring_view_pair.items():
                for value in values:
                    hidden_states_in1.append(norm_hidden_states[:, key])
                    hidden_states_in2.append(norm_hidden_states[:, value])
                    cam_order += [key] * B
            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "concat":
            for key, values in self.neighboring_view_pair.items():
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(torch.cat([
                    norm_hidden_states[:, value] for value in values
                ], dim=1))
                cam_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 2*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "self":
            hidden_states_in1 = rearrange(
                norm_hidden_states, "b n l ... -> b (n l) ...")
            hidden_states_in2 = None
            cam_order = None
        else:
            raise NotImplementedError(
                f"Unknown type: {self.neighboring_attn_type}")
        return hidden_states_in1, hidden_states_in2, cam_order

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        embedding=None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if embedding is not None:
            norm_hidden_states = norm_hidden_states + embedding
        attn_output = self.attn1(
            norm_hidden_states, encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention else None,
            attention_mask=attention_mask, **cross_attention_kwargs,)
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm2(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # multi-view cross attention
        norm_hidden_states = (
            self.norm4(hidden_states, timestep) if self.use_ada_layer_norm else
            self.norm4(hidden_states)
        )
        # batch dim first, cam dim second
        norm_hidden_states = rearrange(
            norm_hidden_states, '(b n) ... -> b n ...', n=self.n_cam)
        B = len(norm_hidden_states)
        # key is query in attention; value is key-value in attention
        hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
            norm_hidden_states, )
        # attention
        attn_raw_output = self.attn4(
            hidden_states_in1,
            encoder_hidden_states=hidden_states_in2,
            **cross_attention_kwargs,
        )
        # final output
        if self.neighboring_attn_type == "self":
            attn_output = rearrange(
                attn_raw_output, 'b (n l) ... -> b n l ...', n=self.n_cam)
        else:
            attn_output = torch.zeros_like(norm_hidden_states)
            for cam_i in range(self.n_cam):
                attn_out_mv = rearrange(attn_raw_output[cam_order == cam_i],
                                        '(n b) ... -> b n ...', b=B)
                attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
        attn_output = rearrange(attn_output, 'b n ... -> (b n) ...')
        # apply zero init connector (one layer)
        attn_output = self.connector(attn_output)
        # short-cut
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (
                1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class TransformerBlockT5(BasicTransformerBlock):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        zero_module_type="zero_linear",
    ):
        super().__init__(
            dim, num_attention_heads, attention_head_dim, dropout,
            cross_attention_dim, activation_fn, num_embeds_ada_norm,
            attention_bias, only_cross_attention, double_self_attention,
            upcast_attention, norm_elementwise_affine, norm_type, final_dropout)
        
        # T5 cross attention
        self.norm4 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        )
        self.attn4 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )

        if zero_module_type == "zero_linear":
            # NOTE: zero_module cannot apply to successive layers.
            self.connector_4 = zero_module(nn.Linear(dim, dim))
        elif zero_module_type == "gated":
            self.connector_4 = GatedConnector(dim)
        elif zero_module_type == "none":
            # TODO: if this block is in controlnet, we may not need zero here.
            self.connector_4 = lambda x: x
        else:
            raise TypeError(f"Unknown zero module type: {zero_module_type}")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_hidden_states_t5=None,
        encoder_attention_mask_t5=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        attn_output = self.attn1(
            norm_hidden_states, encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention else None,
            attention_mask=attention_mask, **cross_attention_kwargs,)
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm2(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
        
        # 3. T5 Cross-Attention
        if self.attn4 is not None:
            norm_hidden_states = (
                self.norm4(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm4(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn4(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states_t5,
                attention_mask=encoder_attention_mask_t5,
                **cross_attention_kwargs,
            )
            # apply zero init connector (one layer)
            attn_output = self.connector_4(attn_output)
            # short-cut
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (
                1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class MultiviewTransformerBlockT5(BasicTransformerBlock):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        # multi_view
        neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
        neighboring_attn_type: Optional[str] = "add",
        zero_module_type="zero_linear",
        attn1_q_trainable=False,
    ):
        super().__init__(
            dim, num_attention_heads, attention_head_dim, dropout,
            cross_attention_dim, activation_fn, num_embeds_ada_norm,
            attention_bias, only_cross_attention, double_self_attention,
            upcast_attention, norm_elementwise_affine, norm_type, final_dropout)
        
        # T5 cross attention
        self.norm4 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        )
        self.attn4 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )

        self.neighboring_view_pair = _ensure_kv_is_int(neighboring_view_pair)
        self.neighboring_attn_type = neighboring_attn_type
        # multiview attention
        self.norm5 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        )
        self.attn5 = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        if zero_module_type == "zero_linear":
            # NOTE: zero_module cannot apply to successive layers.
            self.connector_4 = zero_module(nn.Linear(dim, dim))
            self.connector_5 = zero_module(nn.Linear(dim, dim))
        elif zero_module_type == "gated":
            self.connector_4 = GatedConnector(dim)
            self.connector_5 = GatedConnector(dim)
        elif zero_module_type == "none":
            # TODO: if this block is in controlnet, we may not need zero here.
            self.connector_4 = lambda x: x
            self.connector_5 = lambda x: x
        else:
            raise TypeError(f"Unknown zero module type: {zero_module_type}")
        self.attn1_q_trainable = attn1_q_trainable

    @property
    def new_module(self):
        ret = {
            "norm4": self.norm4,
            "attn4": self.attn4,
            "norm5": self.norm5,
            "attn5": self.attn5,
        }
        if isinstance(self.connector_4, nn.Module):
            ret["connector_4"] = self.connector_4
        if isinstance(self.connector_5, nn.Module):
            ret["connector_5"] = self.connector_5
        if self.attn1_q_trainable:
            ret["attn1.to_q"] = self.attn1.to_q
        return ret

    @property
    def n_cam(self):
        return len(self.neighboring_view_pair)

    def _construct_attn_input(self, norm_hidden_states):
        B = len(norm_hidden_states)
        # reshape, key for origin view, value for ref view
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        if self.neighboring_attn_type == "add":
            for key, values in self.neighboring_view_pair.items():
                for value in values:
                    hidden_states_in1.append(norm_hidden_states[:, key])
                    hidden_states_in2.append(norm_hidden_states[:, value])
                    cam_order += [key] * B
            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "concat":
            for key, values in self.neighboring_view_pair.items():
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(torch.cat([
                    norm_hidden_states[:, value] for value in values
                ], dim=1))
                cam_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 2*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "self":
            hidden_states_in1 = rearrange(
                norm_hidden_states, "b n l ... -> b (n l) ...")
            hidden_states_in2 = None
            cam_order = None
        else:
            raise NotImplementedError(
                f"Unknown type: {self.neighboring_attn_type}")
        return hidden_states_in1, hidden_states_in2, cam_order

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_hidden_states_t5=None,
        encoder_attention_mask_t5=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        embedding=None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if embedding is not None:
            norm_hidden_states = norm_hidden_states + embedding
        attn_output = self.attn1(
            norm_hidden_states, encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention else None,
            attention_mask=attention_mask, **cross_attention_kwargs,)
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm2(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
        
        # 3. T5 Cross-Attention
        if self.attn4 is not None:
            norm_hidden_states = (
                self.norm4(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm4(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn4(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states_t5,
                attention_mask=encoder_attention_mask_t5,
                **cross_attention_kwargs,
            )
            # apply zero init connector (one layer)
            attn_output = self.connector_4(attn_output)
            # short-cut
            hidden_states = attn_output + hidden_states

        # 4. multi-view cross attention
        norm_hidden_states = (
            self.norm5(hidden_states, timestep) if self.use_ada_layer_norm else
            self.norm5(hidden_states)
        )
        # batch dim first, cam dim second
        norm_hidden_states = rearrange(
            norm_hidden_states, '(b n) ... -> b n ...', n=self.n_cam)
        B = len(norm_hidden_states)
        # key is query in attention; value is key-value in attention
        hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
            norm_hidden_states, )
        # attention
        attn_raw_output = self.attn5(
            hidden_states_in1,
            encoder_hidden_states=hidden_states_in2,
            **cross_attention_kwargs,
        )
        # final output
        if self.neighboring_attn_type == "self":
            attn_output = rearrange(
                attn_raw_output, 'b (n l) ... -> b n l ...', n=self.n_cam)
        else:
            attn_output = torch.zeros_like(norm_hidden_states)
            for cam_i in range(self.n_cam):
                attn_out_mv = rearrange(attn_raw_output[cam_order == cam_i],
                                        '(n b) ... -> b n ...', b=B)
                attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
        attn_output = rearrange(attn_output, 'b n ... -> (b n) ...')
        # apply zero init connector (one layer)
        attn_output = self.connector_5(attn_output)
        # short-cut
        hidden_states = attn_output + hidden_states

        # 5. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (
                1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class LearnablePosEmb(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))

    def forward(self, inx):
        return self.param + inx
    

class MotionAttention(nn.Module):
    def __init__(self, in_channels, attnetion_dim, out_channels) -> None:
        super().__init__()

        self.to_qkv = nn.Conv2d(in_channels, attnetion_dim*3, kernel_size=1)
        self.forward_block = nn.Sequential(
            nn.Conv2d(attnetion_dim, attnetion_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(attnetion_dim, attnetion_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.backward_block = nn.Sequential(
            nn.Conv2d(attnetion_dim, attnetion_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(attnetion_dim, attnetion_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.learnable_param = nn.Parameter(torch.ones(2)/2)

    def forward(self, hidden_states):
        B, T = hidden_states.shape[:2] # b, t, c, h, w
        hidden_states_in = self.to_qkv(hidden_states.flatten(0, 1)) # b*t, c, h, w
        hs_q, hs_k, hs_v = torch.chunk(hidden_states_in, 3, dim=1)
        hs_q = hs_q.reshape(B, T, hs_q.shape[1], hs_q.shape[2], hs_q.shape[3])
        hs_k = hs_k.reshape(B, T, hs_k.shape[1], hs_k.shape[2], hs_k.shape[3]) # b, t, c, h, w

        motion_forward = torch.cat([torch.zeros_like(hs_q[:, :1]), hs_q[:, 1:]-hs_k[:, :-1]], dim=1) # b, t, c, h, w
        attn_forward = self.forward_block(motion_forward.flatten(0, 1))

        motion_backward = torch.cat([hs_q[:, :-1]-hs_k[:, 1:], torch.zeros_like(hs_q[:, -1:]), ], dim=1) # b, t, c, h, w
        attn_backward = self.backward_block(motion_backward.flatten(0, 1))

        attn = self.learnable_param[0] * attn_forward + self.learnable_param[1] * attn_backward
        
        outputs = attn * hs_v

        return outputs


class LongShortTemporalMultiviewTransformerBlock(MultiviewTransformerBlockT5):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
            # multi_view
            neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
            neighboring_attn_type: Optional[str] = "add",
            zero_module_type="zero_linear",
            attn1_q_trainable=False,
            # temporal
            video_length=7,
            pos_emb="learnable",
            zero_module_type2="zero_linear",
            spatial_trainable=False,
            # ref_bank
            with_ref=False,
            ref_length=2,
            # can_bus
            with_can_bus=False,
            # motioin
            with_motion=False,
            # attn type
            transformer_type='ff_last',
    ):
        super().__init__(
            dim, num_attention_heads, attention_head_dim, dropout,
            cross_attention_dim, activation_fn, num_embeds_ada_norm,
            attention_bias, only_cross_attention, double_self_attention,
            upcast_attention, norm_elementwise_affine, norm_type, final_dropout,
            neighboring_view_pair, neighboring_attn_type, zero_module_type, attn1_q_trainable)

        self._args = {
            k: v for k, v in locals().items()
            if k != "self" and not k.startswith("_")}
        self.spatial_trainable = spatial_trainable
        self.video_length = video_length
        self.ref_length = ref_length
        self.with_ref = with_ref
        self.with_can_bus = with_can_bus
        self.with_motion = with_motion

        # temporal attn
        if pos_emb == "learnable":
            temp_length = video_length+ref_length if with_ref else video_length
            self.pos_emb = LearnablePosEmb(size=(1, temp_length, dim))
        elif pos_emb == "none":
            self.pos_emb = None
        else:
            raise NotImplementedError(f"Unknown type {pos_emb}")

        if self.use_ada_layer_norm:
            self.temp_norm = AdaLayerNorm(dim, num_embeds_ada_norm)
        else:
            self.temp_norm = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine)
        self.temp_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.temp_connector = get_zero_module(zero_module_type2, dim)

        if with_ref:
            self.ref_linear = nn.Linear(dim, dim, bias=True)
        if with_can_bus:
            self.can_bus_linear = nn.Linear(cross_attention_dim, dim, bias=True)
        if with_motion:
            self.motion_attn = MotionAttention(dim, dim, dim)
            self.motion_connector = get_zero_module(zero_module_type2, dim)
        self.transformer_type = transformer_type
        self._sc_attn_index = None

    @property
    def new_module(self):
        if self.spatial_trainable:
            ret = {
                "norm4": self.norm4,
                "attn4": self.attn4,
                "norm5": self.norm5,
                "attn5": self.attn5,
            }
            if isinstance(self.connector_4, nn.Module):
                ret["connector_4"] = self.connector_4
            if isinstance(self.connector_5, nn.Module):
                ret["connector_5"] = self.connector_5
        else:
            ret = {}
        ret = {
            "temp_attn": self.temp_attn,
            "temp_norm": self.temp_norm,
            **ret,
        }

        if self.with_ref:
            ret['ref_linear'] = self.ref_linear

        if self.with_can_bus:
            ret['can_bus_linear'] = self.can_bus_linear
        
        if self.with_motion:
            ret['motion_attn'] = self.motion_attn
            if isinstance(self.motion_connector, nn.Module):
                ret['motion_connector'] = self.motion_connector

        if isinstance(self.temp_connector, nn.Module):
            ret["temp_connector"] = self.temp_connector
        if isinstance(self.pos_emb, nn.Module):
            ret["temp_pos_emb"] = self.pos_emb
        
        if self.transformer_type.startswith("_"):
            ret['attn1.to_q'] = self.attn1.to_q
        
        return ret

    @property
    def sc_attn_index(self):
        # one can set `self._sc_attn_index` to a function for convenient changes
        # among batches.
        if callable(self._sc_attn_index):
            return self._sc_attn_index()
        else:
            return self._sc_attn_index

    def _construct_sc_attn_input(
        self, norm_hidden_states, sc_attn_index, type="add"
    ):
        # assume data has form (b, frame, c), frame == len(sc_attn_index)
        # return two sets of hidden_states and an order list.
        B = len(norm_hidden_states)
        hidden_states_in1 = []
        hidden_states_in2 = []
        back_order = []

        if type == "add":
            for key, values in zip(range(len(sc_attn_index)), sc_attn_index):
                for value in values:
                    hidden_states_in1.append(norm_hidden_states[:, key])
                    hidden_states_in2.append(norm_hidden_states[:, value])
                    back_order += [key] * B
            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            back_order = torch.LongTensor(back_order)
        elif type == "concat":
            for key, values in zip(range(len(sc_attn_index)), sc_attn_index):
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(torch.cat([
                    norm_hidden_states[:, value] for value in values
                ], dim=1))
                back_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 3*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            back_order = torch.LongTensor(back_order)
        else:
            raise NotImplementedError(f"Unknown type: {type}")
        return hidden_states_in1, hidden_states_in2, back_order
    
    def forward_old_attns(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_hidden_states_t5=None,
        encoder_attention_mask_t5=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        embedding=None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        shift_mlp, scale_mlp, gate_mlp = None, None, None
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if embedding is not None:
            norm_hidden_states = norm_hidden_states + embedding
        if self.transformer_type.startswith("_"):
            norm_hidden_states = rearrange(
                norm_hidden_states, "(b f n) d c -> (b n) f d c",
                f=self.video_length, n=self.n_cam)
            B = len(norm_hidden_states)

            # this index is for kv pair, your dataloader should make it consistent.
            norm_hidden_states_q, norm_hidden_states_kv, back_order = self._construct_sc_attn_input(
                norm_hidden_states, self.sc_attn_index, type="concat")

            attn_raw_output = self.attn1(
                norm_hidden_states_q,
                encoder_hidden_states=norm_hidden_states_kv,
                attention_mask=attention_mask, **cross_attention_kwargs)
            attn_output = torch.zeros_like(norm_hidden_states)
            for frame_i in range(self.video_length):
                # TODO: any problem here? n should == 1
                attn_out_mt = rearrange(attn_raw_output[back_order == frame_i],
                                        '(n b) ... -> b n ...', b=B)
                attn_output[:, frame_i] = torch.sum(attn_out_mt, dim=1)
            attn_output = rearrange(
                attn_output, "(b n) f d c -> (b f n) d c", n=self.n_cam)
        else:
            attn_output = self.attn1(
                norm_hidden_states, encoder_hidden_states=encoder_hidden_states
                if self.only_cross_attention else None,
                attention_mask=attention_mask, **cross_attention_kwargs,)
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm2(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. T5 Cross-Attention
        if self.attn4 is not None:
            norm_hidden_states = (
                self.norm4(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm4(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn4(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states_t5,
                attention_mask=encoder_attention_mask_t5,
                **cross_attention_kwargs,
            )
            # apply zero init connector (one layer)
            attn_output = self.connector_4(attn_output)
            # short-cut
            hidden_states = attn_output + hidden_states

        # 4. multi-view cross attention
        norm_hidden_states = (
            self.norm5(hidden_states, timestep) if self.use_ada_layer_norm else
            self.norm5(hidden_states)
        )
        # batch dim first, cam dim second
        norm_hidden_states = rearrange(
            norm_hidden_states, '(b n) ... -> b n ...', n=self.n_cam)
        B = len(norm_hidden_states)
        # key is query in attention; value is key-value in attention
        hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
            norm_hidden_states, )
        # attention
        bs1, dim1 = hidden_states_in1.shape[:2]
        grpn = 6  # TODO: hard-coded to use bs=6, avoiding numerical error.
        if bs1 > grpn and dim1 > 1400 and not is_xformers(self.attn5):
            hidden_states_in1s = torch.split(hidden_states_in1, grpn)
            hidden_states_in2s = torch.split(hidden_states_in2, grpn)
            grps = len(hidden_states_in1s)
            attn_raw_output = [None for _ in range(grps)]
            for i in range(grps):
                attn_raw_output[i] = self.attn5(
                    hidden_states_in1s[i],
                    encoder_hidden_states=hidden_states_in2s[i],
                    **cross_attention_kwargs,
                )
            attn_raw_output = torch.cat(attn_raw_output, dim=0)
        else:
            attn_raw_output = self.attn5(
                hidden_states_in1,
                encoder_hidden_states=hidden_states_in2,
                **cross_attention_kwargs,
            )
        # final output
        if self.neighboring_attn_type == "self":
            attn_output = rearrange(
                attn_raw_output, 'b (n l) ... -> b n l ...', n=self.n_cam)
        else:
            attn_output = torch.zeros_like(norm_hidden_states)
            for cam_i in range(self.n_cam):
                attn_out_mv = rearrange(attn_raw_output[cam_order == cam_i],
                                        '(n b) ... -> b n ...', b=B)
                attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
        attn_output = rearrange(attn_output, 'b n ... -> (b n) ...')
        # apply zero init connector (one layer)
        attn_output = self.connector_5(attn_output)
        # short-cut
        hidden_states = attn_output + hidden_states

        return hidden_states, shift_mlp, scale_mlp, gate_mlp
    
    def add_temp_pos_emb(self, hidden_state):
        if self.pos_emb is None:
            return hidden_state
        return self.pos_emb(hidden_state)

    def forward_temporal(self, hidden_states, timestep):       
        if self.with_ref:
            bank_fea = [self.ref_linear(rearrange(self.ref_hidden_states.clone(), "(b f n) d c -> (b d n) f c",
                f=self.ref_length, n=self.n_cam))]
        else:
            bank_fea = []

        # Temporal-Attention
        d = hidden_states.shape[1]
        hidden_states_in = rearrange(
            hidden_states, "(b f n) d c -> (b d n) f c", f=self.video_length,
            n=self.n_cam)
        hidden_states_in = torch.cat(bank_fea + [hidden_states_in], dim=1)
        hidden_states_in = self.add_temp_pos_emb(hidden_states_in)
        if self.with_can_bus:
            can_bus_embedding = self.can_bus_embedding.repeat(1, d, 1)
            can_bus_embedding = rearrange(
                can_bus_embedding, "(b f n) d c -> (b d n) f c",
                f=self.ref_length+self.video_length, n=self.n_cam
            )
            hidden_states_in = hidden_states_in + can_bus_embedding
        norm_hidden_states = (
            self.temp_norm(hidden_states_in, timestep)
            if self.use_ada_layer_norm else self.temp_norm(hidden_states_in))
        norm_hidden_states_ = norm_hidden_states
        # NOTE: xformers cannot take bs larger than 8192
        if len(norm_hidden_states) >= 8192:
            chunk_num = math.ceil(len(norm_hidden_states) / 4096.)
            norm_hidden_states = norm_hidden_states.chunk(chunk_num)
            attn_output = torch.cat([
                self.temp_attn(norm_hidden_states[i]) for i in range(chunk_num)
            ], dim=0)
        else:
            attn_output = self.temp_attn(norm_hidden_states)
        attn_output = attn_output[:, self.ref_length:]
        # apply zero init connector (one layer)
        attn_output = self.temp_connector(attn_output)
        attn_output = rearrange(
            attn_output, "(b d n) f c -> (b f n) d c", d=d, n=self.n_cam)
        
        # short-term motion attn
        if self.with_motion:
            h, w = self.size
            norm_hidden_states = rearrange(
                norm_hidden_states_[:, self.ref_length:], "(b d n) f c -> (b n) f c d", d=d, n=self.n_cam)
            norm_hidden_states = rearrange(
                norm_hidden_states, "b f c (h w) -> b f c h w", h=h, w=w
            )
            motion_output = self.motion_attn(norm_hidden_states)
            motion_output = self.motion_connector(rearrange(
                motion_output, "(b n f) c h w -> (b f n) (h w) c", f=self.video_length, n=self.n_cam, h=h, w=w))

            # short-cut
            hidden_states = attn_output + hidden_states + motion_output
        else:
            # short-cut
            hidden_states = attn_output + hidden_states

        return hidden_states

    def forward_ff_last(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_hidden_states_t5=None,
        encoder_attention_mask_t5=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        embedding=None,
    ):
        hidden_states, shift_mlp, scale_mlp, gate_mlp = self.forward_old_attns(
            hidden_states, attention_mask, encoder_hidden_states,
            encoder_attention_mask, encoder_hidden_states_t5, encoder_attention_mask_t5, 
            timestep, cross_attention_kwargs, class_labels, embedding=embedding
        )

        hidden_states = self.forward_temporal(hidden_states, timestep)

        # 5. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (
                1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states
    
        return hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_hidden_states_t5=None,
        encoder_attention_mask_t5=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        size=None,
        scene_embedding=None,
        can_bus_embedding=None,
    ):   
        self.size = size
        if self.with_can_bus and can_bus_embedding is not None:
            self.can_bus_embedding = self.can_bus_linear(can_bus_embedding)
        if self.with_ref:
            hidden_states = rearrange(
                    hidden_states, "(b f n) d c -> (b n) f d c",
                    f=self.ref_length+self.video_length, n=self.n_cam
                )
            self.ref_hidden_states = rearrange(
                    hidden_states[:, :self.ref_length].clone(), "(b n) f d c -> (b f n) d c",
                    f=self.ref_length, n=self.n_cam
                )
            hidden_states = rearrange(
                    hidden_states[:, self.ref_length:].clone(), "(b n) f d c -> (b f n) d c",
                    f=self.video_length, n=self.n_cam
                )

        hidden_states = self.forward_ff_last(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_t5,
            encoder_attention_mask_t5,
            timestep,
            cross_attention_kwargs,
            class_labels,
            embedding=scene_embedding
        )

        if self.with_ref:
            ref_hidden_states = rearrange(
                    self.ref_hidden_states, "(b f n) d c -> (b n) f d c",
                    f=self.ref_length, n=self.n_cam
                )
            hidden_states = rearrange(
                    hidden_states, "(b f n) d c -> (b n) f d c",
                    f=self.video_length, n=self.n_cam
                )
            hidden_states = torch.cat([ref_hidden_states, hidden_states], dim=1)
            hidden_states = rearrange(
                    hidden_states, "(b n) f d c -> (b f n) d c",
                    f=self.ref_length+self.video_length, n=self.n_cam
                )

        return hidden_states


class Transformer2DModelT5(Transformer2DModel):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        ## scene position embedding ##
        scene_channels: int=320,
        attn1_q_trainable: bool=False,
        temporal_attention: bool = False,
    ):
        super().__init__(
            num_attention_heads, attention_head_dim, in_channels, out_channels,
            num_layers, dropout, norm_num_groups, cross_attention_dim,
            attention_bias, sample_size, num_vector_embeds, patch_size,
            activation_fn, num_embeds_ada_norm, use_linear_projection,
            only_cross_attention, upcast_attention, norm_type, norm_elementwise_affine
        )

        self.temporal_attention = temporal_attention

        ## scene position embedding ##
        self.bank = []
        inner_dim = num_attention_heads * attention_head_dim
        self.scene_proj_in = nn.Conv2d(scene_channels, inner_dim, kernel_size=3, stride=1, padding=1)
        ##################

    @property
    def new_module(self):
        ret = {
            'scene_proj_in': self.scene_proj_in
        }
        for name, mod in list(self.named_modules()):
            if isinstance(mod, BasicMultiviewTransformerBlock):
                for k, v in _get_module(self, name).new_module.items():
                    ret[f"{name}.{k}"] = v
        return ret

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_t5: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask_t5: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            encoder_attention_mask ( `torch.Tensor`, *optional* ).
                Cross-attention mask, applied to encoder_hidden_states. Two formats supported:
                    Mask `(batch, sequence_length)` True = keep, False = discard. Bias `(batch, 1, sequence_length)` 0
                    = keep, -10000 = discard.
                If ndim == 2: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            ## scene position embedding ##
            if len(self.bank) != 0:
                embedding = F.interpolate(self.bank[0], size=(height, width), mode='bilinear', align_corners=False)
                embedding = self.scene_proj_in(embedding).permute(0, 2, 3, 1).flatten(1, 2)
            else:
                embedding = None
            ##################

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = self.proj_in(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.temporal_attention and embedding is not None:   # temporal + multiview + T5 attention
                # unet w/ embedding and temporal cross-view attention using LongShortTemporalMultiviewTransformerBlock
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_t5=encoder_hidden_states_t5,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_attention_mask_t5=encoder_attention_mask_t5,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    size=[height, width],
                    scene_embedding=embedding,
                    can_bus_embedding=self.bank[-1],
                )
            elif embedding is not None: # multiview + T5 attention
                # unet w/ embedding and cross-view attention using BasicMultiviewTransformerBlock
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_t5=encoder_hidden_states_t5,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_attention_mask_t5=encoder_attention_mask_t5,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    embedding=embedding,
                )
            else:
                # controlnet w/o embedding and cross-view attention using BasicTransformerBlock
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_t5=encoder_hidden_states_t5,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_attention_mask_t5=encoder_attention_mask_t5,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = self.proj_out(hidden_states)
            else:
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()
        elif self.is_input_patches:
            # TODO: cleanup!
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)

            # unpatchify
            height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class CrossAttnDownBlock2DT5(CrossAttnDownBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__(
            in_channels, out_channels, temb_channels, dropout, num_layers,
            resnet_eps, resnet_time_scale_shift, resnet_act_fn, resnet_groups,
            resnet_pre_norm, attn_num_head_channels, cross_attention_dim,
            output_scale_factor, downsample_padding, add_downsample,
            dual_cross_attention, use_linear_projection, only_cross_attention,
            upcast_attention,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states_t5: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask_t5: Optional[torch.FloatTensor] = None,
    ):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_t5,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    encoder_attention_mask_t5,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_hidden_states_t5=encoder_hidden_states_t5,
                    encoder_attention_mask_t5=encoder_attention_mask_t5,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock2DT5(CrossAttnUpBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__(
            in_channels, out_channels, prev_output_channel, temb_channels,
            dropout, num_layers, resnet_eps, resnet_time_scale_shift,
            resnet_act_fn, resnet_groups, resnet_pre_norm,
            attn_num_head_channels, cross_attention_dim, output_scale_factor,
            add_upsample, dual_cross_attention, use_linear_projection,
            only_cross_attention, upcast_attention,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states_t5: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask_t5: Optional[torch.FloatTensor] = None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_t5,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    encoder_attention_mask_t5,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_t5=encoder_hidden_states_t5,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_attention_mask_t5=encoder_attention_mask_t5,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UNetMidBlock2DCrossAttnT5(UNetMidBlock2DCrossAttn):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
    ):
        super().__init__(
            in_channels, temb_channels, dropout, num_layers, resnet_eps,
            resnet_time_scale_shift, resnet_act_fn, resnet_groups,
            resnet_pre_norm, attn_num_head_channels, output_scale_factor,
            cross_attention_dim, dual_cross_attention, use_linear_projection,
            upcast_attention,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states_t5: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask_t5: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_t5=encoder_hidden_states_t5,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                encoder_attention_mask_t5=encoder_attention_mask_t5,
                return_dict=False,
            )[0]
            hidden_states = resnet(hidden_states, temb)

        return hidden_states