from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.controlnet import zero_module
from xscene.networks.blocks import TriplaneConv, TriplaneSiLU, TriplaneDownsample2x

class TriplaneMapConditioningEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int = 64,
        conditioning_size: Tuple[int, int, int] = (16, 216, 216),  # only use 25
        block_out_channels: Tuple[int] = (32, 64, 128, 256),
        is_rollout: bool = True,
        downsample: bool = True,
        conv_down: bool = True,
        tri_z_down: bool = False,
    ):
        super().__init__()
        # input size   25, 216
        # output size   C, 116, 116

        self.conv_in = nn.Sequential(
            TriplaneConv(
                conditioning_size[0],
                block_out_channels[0],
                kernel_size=3, padding=1, is_rollout=is_rollout),
            TriplaneSiLU(),
        )

        self.blocks = nn.ModuleList([])
        if downsample:
            self.blocks.append(nn.Sequential(
                TriplaneConv(
                    block_out_channels[0], block_out_channels[0], kernel_size=3, padding=1, is_rollout=is_rollout),
                TriplaneSiLU(),
                TriplaneDownsample2x(
                    block_out_channels[0], block_out_channels[1], conv_down=conv_down, tri_z_down=tri_z_down),
                TriplaneSiLU(),
            ))
        else:
            self.blocks.append(nn.Sequential(
                TriplaneConv(
                    block_out_channels[0], block_out_channels[1], kernel_size=3, padding=1, is_rollout=is_rollout),
                TriplaneSiLU(),
            ))

        for i in range(1, len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Sequential(
                TriplaneConv(
                    channel_in, channel_out, kernel_size=3, padding=1, is_rollout=is_rollout),
                TriplaneSiLU()
            ))

        self.conv_out = zero_module(
            TriplaneConv(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
                is_rollout=is_rollout,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)

        for block in self.blocks:
            embedding = block(embedding)

        embedding = self.conv_out(embedding)

        return embedding

class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        output_size = None,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.output_size = output_size        
        if output_size is not None:
            self.up_conv = nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1)
            self.connector = zero_module(nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        if self.output_size is not None:
            embedding = F.interpolate(embedding, self.output_size, mode='bilinear', align_corners=False)
            embedding = embedding + self.connector(F.silu(self.up_conv(embedding)))

        embedding = self.conv_out(embedding)

        return embedding


class BEVControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int = 320,
        conditioning_size: Tuple[int, int, int] = (25, 200, 200),  # only use 25
        block_out_channels: Tuple[int] = (32, 64, 128, 256),
        output_size = None,
    ):
        super().__init__()
        # input size   25, 200, 200
        # output size 320,  28,  50

        self.conv_in = nn.Conv2d(
            conditioning_size[0],
            block_out_channels[0],
            kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 2):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                nn.Conv2d(
                    channel_in, channel_out, kernel_size=3, padding=(2, 1),
                    stride=2))
        channel_in = block_out_channels[-2]
        channel_out = block_out_channels[-1]
        self.blocks.append(
            nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=(2, 1))
        )
        self.blocks.append(
            nn.Conv2d(
                channel_in, channel_out, kernel_size=3, padding=(2, 1),
                stride=(2, 1)))

        self.output_size = output_size        
        if output_size is not None:
            self.up_conv = nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1)
            self.connector = zero_module(nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1))

        self.conv_out = zero_module(
            nn.Conv2d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        if self.output_size is not None:
            embedding = F.interpolate(embedding, self.output_size, mode='bilinear', align_corners=False)
            embedding = embedding + self.connector(F.silu(self.up_conv(embedding)))

        embedding = self.conv_out(embedding)

        return embedding


class BEVControlNetConditioningEmbeddingPlus(BEVControlNetConditioningEmbedding):
    def __init__(
        self,
        conditioning_embedding_size: Tuple[int],
        conditioning_embedding_channels: int = 320,
        conditioning_size: Tuple[int, int, int] = (25, 200, 200),  # only use 25
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        super().__init__()
        # input size   25, 200, 200
        # output size 320,  32,  88

        self.conv_in = nn.Conv2d(
            conditioning_size[0], block_out_channels[0], kernel_size=3,
            padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 2):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=(1, 1))
            )
            stride = 1 if i == 0 else 2
            self.blocks.append(
                nn.Conv2d(
                    channel_in, channel_out, kernel_size=3, padding=(1, 1),
                    stride=stride))
        channel_in = block_out_channels[-2]
        channel_out = block_out_channels[-1]
        self.blocks.append(
            nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=(1, 1))
        )
        self.blocks.append(
            nn.Conv2d(
                channel_in, channel_out, kernel_size=3, padding=(1, 1),
                stride=(2, 1)))
        self.blocks.append(nn.AdaptiveAvgPool2d(conditioning_embedding_size))

        self.conv_out = zero_module(
            nn.Conv2d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )
