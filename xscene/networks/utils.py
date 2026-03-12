import torch
import torch.nn as nn
import numpy as np
from functools import lru_cache

def compose_featmaps(feat_xy, feat_xz, feat_yz, transpose=True):
    tri_size = {
        200: (200, 200, 16),
        100: (100, 100, 16),
        50: (50, 50, 16),
        25: (25, 25, 16),
        13: (13, 13, 16),
    }
    H, W, D = tri_size[feat_xy.shape[-1]]

    empty_block = torch.zeros(list(feat_xy.shape[:-2]) + [D, D], dtype=feat_xy.dtype, device=feat_xy.device)
    if transpose:
        feat_yz = feat_yz.transpose(-1, -2)
    composed_map = torch.cat(                          
        [torch.cat([feat_xy, feat_xz], dim=-1),         # B,C,X,Y+Z 
         torch.cat([feat_yz, empty_block], dim=-1)],    # B,C,Z,Y+Z 
        dim=-2                                          # B,C,X+Z,Y+Z
    )
    return composed_map


def decompose_featmaps(composed_map, transpose=True):
    tri_size = {
        216: (200, 200, 16),
        116: (100, 100, 16),
        66: (50, 50, 16),
        41: (25, 25, 16),
        29: (13, 13, 16),
    }
    H, W, D = tri_size[composed_map.shape[-1]]
    feat_xy = composed_map[..., :H, :W] # (..., X, Y)
    feat_xz = composed_map[..., :H, W:] # (..., X, Z)
    feat_yz = composed_map[..., H:, :W] # (..., Z, Y)
    if transpose:
        return feat_xy, feat_xz, feat_yz.transpose(-1, -2)
    else:
        return feat_xy, feat_xz, feat_yz

@lru_cache(4)
def voxel_coord(voxel_shape):
    x = np.arange(voxel_shape[0])
    y = np.arange(voxel_shape[1])
    z = np.arange(voxel_shape[2])
    X, Y, Z = np.meshgrid(x, y, z)
    voxel_coord = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)
    return voxel_coord


def make_query(grid_size):
    coords = torch.from_numpy(voxel_coord(tuple(grid_size)))
    coords = coords.reshape(-1, 3)
    query = torch.zeros(coords.shape, dtype=torch.float32)
    query[:,0] = 2*coords[:,0].clamp(0,grid_size[0]-1)/float(grid_size[0]-1) -1
    query[:,1] = 2*coords[:,1].clamp(0,grid_size[1]-1)/float(grid_size[1]-1) -1
    query[:,2] = 2*coords[:,2].clamp(0,grid_size[2]-1)/float(grid_size[2]-1) -1
    
    query = query.reshape(-1, 3)
    return coords.unsqueeze(0), query.unsqueeze(0)


def point2voxel(grid_size, preds, coords):
    output = torch.zeros((preds.shape[0], grid_size[0], grid_size[1], grid_size[2]), device=preds.device, dtype=preds.dtype)
    for i in range(preds.shape[0]):
        output[i, coords[i, :, 0], coords[i, :, 1], coords[i, :, 2]] = preds[i]
    return output


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(groups, channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(groups, channels)

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)