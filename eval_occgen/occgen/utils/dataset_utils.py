from enum import Enum

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


class Command(Enum):
    STATIC = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3


def get_subsequence_command(traj_rot, angle_thr, forward_thr):
    rot_diff = traj_rot[:, 2] - traj_rot[0, 2]
    rot_diff = (rot_diff + np.pi) % (2 * np.pi) - np.pi
    if np.any(rot_diff > angle_thr):
        return Command.LEFT
    elif np.any(rot_diff < -angle_thr):
        return Command.RIGHT

    displacement = np.sqrt((traj_rot[:, 0] - traj_rot[0, 0]) ** 2 + (traj_rot[:, 1] - traj_rot[0, 1]) ** 2)
    if np.any(displacement > forward_thr):
        return Command.FORWARD
    else:
        return Command.STATIC


def get_layout(voxels, down_size):
    voxel_binary = (voxels == 10).to(torch.float32)  # T, X, Y, Z
    bev = voxel_binary.max(dim=3).values  # T, X, Y
    bev = bev.reshape(voxels.shape[0], 1, bev.shape[1], bev.shape[2])  # T, 1, X, Y
    layout = torch.nn.functional.max_pool2d(bev, down_size)  # T, 1, X, Y
    return layout[:, 0]  # T, X, Y


def parse_hexplane_path(dataset_cfg, voxel_path):
    if dataset_cfg.dataset == 'carlasc':
        return (voxel_path
                .replace(dataset_cfg.data_path.rstrip('/') + '/', '')
                .replace(dataset_cfg.scene_folder.rstrip('/') + '/', '')
                .replace('.label', '.npy'))
    if dataset_cfg.dataset == 'occ3dn':
        return voxel_path.replace('.npz', '.npy')
    if dataset_cfg.dataset == 'occ3dw':
        return (voxel_path
                .replace(dataset_cfg.data_path.rstrip('/') + '/', '')
                .replace('.npz', '.npy'))
    raise NotImplementedError


def get_traj_rot(poses):
    return np.concatenate((get_trajectory(poses), get_angles(poses)[..., None]), axis=-1)


def get_trajectory(poses):
    return poses[:, [3, 7]] - poses[0, [3, 7]]


def get_angles(poses):
    angles = np.arctan2(poses[:, 4], poses[:, 0])
    return angles


def get_traj_rot_from_coordinates(coordinates):
    trajectory = coordinates[:, :2] - coordinates[0, :2]
    dx = coordinates[1:, 0] - coordinates[:-1, 0]
    dy = coordinates[1:, 1] - coordinates[:-1, 1]
    angles = np.concatenate(([0], np.arctan2(dy, dx)))
    return np.concatenate((trajectory, angles[:, None]), axis=-1)


def get_trajectory_from_coordinates(coordinates):
    return (coordinates - coordinates[0])[..., :2]


def apply_augmentation(voxels, invalids, paths, aug_type=0):
    if aug_type & 1:  # Flip X
        voxels = torch.flip(voxels, dims=[1])
        invalids = torch.flip(invalids, dims=[1])
    if aug_type & 2:  # Flip Y
        voxels = torch.flip(voxels, dims=[2])
        invalids = torch.flip(invalids, dims=[2])
    if aug_type & 4:  # Flip T
        voxels = torch.flip(voxels, dims=[0])
        invalids = torch.flip(invalids, dims=[0])
        paths = paths[::-1]
    return voxels, invalids, paths


def compute_tdf(voxel_label: np.ndarray, trunc_distance: float = 3, trunc_value: float = -1) -> np.ndarray:
    """ Compute Truncated Distance Field (TDF). voxel_label -- [X, Y, Z] """
    # make TDF at free voxels.
    # distance is defined as Euclidean distance to nearest unfree voxel (occupied or unknown).
    free = voxel_label == 0
    tdf = distance_transform_edt(free)
    # Set -1 if distance is greater than truncation_distance
    tdf[tdf > trunc_distance] = trunc_value
    return tdf


def get_query(voxel, num_classes, grid_size, max_points):
    num_points = 0
    query_gt, query_int = list(), list()

    # add all non-free voxels to query
    for i in range(1, num_classes):
        class_occupancy = torch.tensor(voxel == i)
        query_int_class = torch.nonzero(class_occupancy)
        num_points += query_int_class.shape[0]
        query_int.append(query_int_class)
        query_gt.append(torch.zeros(query_int_class.shape[0], dtype=torch.int64) + i)

    # add voxels near occupied voxels
    tdf = compute_tdf(voxel, trunc_distance=2)
    query_int_tdf = torch.nonzero(torch.tensor(np.logical_and(tdf > 0, tdf <= 2)))
    query_gt_tdf = torch.zeros(query_int_tdf.shape[0], dtype=torch.int64)
    num_points += query_int_tdf.shape[0]
    query_int.append(query_int_tdf)
    query_gt.append(query_gt_tdf)

    # randomly add some far & free points
    num_free_points = max_points - num_points
    if num_free_points > 0:
        query_int_free = torch.nonzero(torch.tensor(np.logical_and(voxel == 0, tdf == -1)), as_tuple=False)
        query_int_free_i = torch.randperm(query_int_free.shape[0])
        query_int_free = query_int_free[query_int_free_i][: min(query_int_free.shape[0], num_free_points)]
        query_gt_free = torch.zeros(query_int_free.shape[0], dtype=torch.int64)
        num_points += query_int_free.shape[0]
        query_int.append(query_int_free)
        query_gt.append(query_gt_free)

    assert num_points >= max_points

    query_int = torch.cat(query_int, dim=0)[:max_points]
    query_gt = torch.cat(query_gt, dim=0)[:max_points]

    query = torch.zeros_like(query_int).to(torch.float32)
    query[:, 0] = 2 * query_int[:, 0].clamp(0, grid_size[0] - 1) / float(grid_size[0] - 1) - 1
    query[:, 1] = 2 * query_int[:, 1].clamp(0, grid_size[1] - 1) / float(grid_size[1] - 1) - 1
    query[:, 2] = 2 * query_int[:, 2].clamp(0, grid_size[2] - 1) / float(grid_size[2] - 1) - 1

    return query, query_gt, query_int
