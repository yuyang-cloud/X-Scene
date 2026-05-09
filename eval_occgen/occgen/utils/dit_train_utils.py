from collections import OrderedDict

import math
import numpy as np
import torch
from PIL import Image
from torch.utils.data.distributed import DistributedSampler


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class DistributedTurningSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, repeat_factor=2, shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.repeat_factor = repeat_factor
        self.indices = self._get_indices()
        if self.drop_last and len(self.indices) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.indices) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def _get_indices(self):
        if self.repeat_factor == 1:
            return list(range(len(self.dataset)))
        turning_indices, non_turning_indices = [], []
        for i, data in enumerate(self.dataset):
            if data['turn']:
                turning_indices.append(i)
            else:
                non_turning_indices.append(i)
        indices = turning_indices * self.repeat_factor + non_turning_indices
        return indices

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.indices), generator=g).tolist()
            indices = [self.indices[i] for i in indices]
        else:
            indices = self.indices.copy()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
