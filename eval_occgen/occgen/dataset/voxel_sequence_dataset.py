import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from occgen.utils.dataset_utils import apply_augmentation


class VoxelSequenceDataset(ABC, Dataset):
    def __init__(self, dataset_cfg, imageset='train', force_length=-1, force_shuffle=False):
        self.dataset_cfg = dataset_cfg
        self.imageset = imageset

        self.learning_map = np.array(list(dataset_cfg.learning_map.values()))
        self.num_classes = dataset_cfg.num_classes
        self.sequence_length = dataset_cfg.t_length
        self.grid_size = dataset_cfg.grid_size
        self.train_aug = dataset_cfg.train_aug
        self.sequences = self.get_sequence()  # OVERRIDE!

        if force_shuffle:
            random.shuffle(self.sequences)

        if force_length > 0:
            self.sequences = self.sequences[:force_length]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        enable_aug = self.imageset == 'train' and random.random() < 0.5 and self.train_aug  # 50% chance of aug
        aug_type = random.randint(0, 7) if enable_aug else 0
        return self.get_item_with_aug(index, aug_type=aug_type)

    def get_item_with_aug(self, index, aug_type=0):
        voxels, invalids, paths = list(), list(), list()

        for i, path in enumerate(self.sequences[index]):
            voxel, invalid = self.get_voxel(path)  # OVERRIDE!
            voxels.append(voxel)
            invalids.append(invalid)
            paths.append(path)

        # from pathlib import Path
        #
        # if 'Heavy' in str(paths[0]):
        #     voxels[0].tofile(Path('out/voxel/carla_gt') / Path(paths[0]).parts[3] / Path(paths[0]).name)

        voxels, invalids, paths = [default_collate(arr) for arr in [voxels, invalids, paths]]
        voxels, invalids, paths = apply_augmentation(voxels, invalids, paths, aug_type)

        return {
            'voxel': voxels,
            'invalid': invalids,
            'paths': paths,
        }

    @abstractmethod
    def get_sequence(self):
        pass

    @abstractmethod
    def get_voxel(self, path):
        pass


class VoxelSequenceQueryDataset(VoxelSequenceDataset, ABC):
    def __init__(self, dataset_cfg, imageset='train', force_length=-1, force_shuffle=False):
        super().__init__(dataset_cfg, imageset, force_length, force_shuffle)

    def __getitem__(self, index):
        voxels, queries, query_gts, queries_int, invalids, paths = [list() for _ in range(6)]

        for i, path in enumerate(self.sequences[index]):
            voxel, query, query_gt, query_int, invalid = self.get_voxel(path)  # OVERRIDE!
            voxels.append(voxel)
            invalids.append(invalid)
            paths.append(path)

            time_col = torch.zeros(query.shape[0], 1)
            query_offset = 2 * i / (len(self.sequences[index]) - 1) - 1
            query = torch.cat([query, time_col + query_offset], dim=1)
            query_int = torch.cat([query_int, time_col + i], dim=1).int()
            queries.append(query)
            query_gts.append(query_gt)
            queries_int.append(query_int)

        voxels, invalids, paths = [default_collate(arr) for arr in [voxels, invalids, paths]]

        query_gts = default_collate(query_gts)
        queries, queries_int = [torch.cat(arr) for arr in [queries, queries_int]]

        return {
            'voxel': voxels,
            'invalid': invalids,
            'paths': paths,
            'queries': queries,
            'queries_int': queries_int,
            'query_gts': query_gts,
        }


class VoxelDataset(ABC, Dataset):
    def __init__(self, dataset_cfg, imageset='train'):
        self.dataset_cfg = dataset_cfg
        self.imageset = imageset

        self.learning_map = np.array(list(dataset_cfg.learning_map.values()))
        self.num_classes = dataset_cfg.num_classes
        self.grid_size = dataset_cfg.grid_size

        scenes = self.get_scenes()
        if isinstance(scenes, tuple):
            self.scenes, self.lidar2ego_list = scenes
        else:
            self.scenes = scenes
            self.lidar2ego_list = None

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        path = self.scenes[index]
        if self.lidar2ego_list is None:
            voxel, invalid = self.get_voxel(path)
        else:
            lidar2ego = self.lidar2ego_list[index]
            voxel, invalid = self.get_voxel(path, lidar2ego)
        return {
            'voxel': voxel,
            'invalid': invalid,
            'path': path,
        }

    @abstractmethod
    def get_scenes(self):
        pass

    @abstractmethod
    def get_voxel(self, path):
        pass
