import pickle
import random
from pathlib import Path

import numpy as np

from occgen.dataset.hexplane_dataset import HexplaneDataset
from occgen.dataset.voxel_sequence_dataset import VoxelDataset, VoxelSequenceDataset, VoxelSequenceQueryDataset
from occgen.utils.dataset_utils import get_query, get_traj_rot


class Occ3DW(VoxelSequenceDataset):
    def __init__(self, dataset_cfg, imageset='train', force_length=-1, force_shuffle=False):
        super().__init__(dataset_cfg, imageset, force_length, force_shuffle)

        map_size = max(dataset_cfg.learning_map.keys()) + 1
        self.learning_map = np.zeros(map_size, dtype=int)
        keys = np.array(list(dataset_cfg.learning_map.keys()))
        values = np.array(list(dataset_cfg.learning_map.values()))
        self.learning_map[keys] = values

    def get_sequence(self):
        sequences = list()

        data_folder = Path(self.dataset_cfg.data_path) / self.dataset_cfg.imageset[self.imageset]
        scene_roots = list(data_folder.iterdir())

        if self.dataset_cfg.get('subset_percentage', None) is not None:
            num_of_scenes = int(len(scene_roots) * self.dataset_cfg.subset_percentage)
            scene_roots = scene_roots[:num_of_scenes]

        actual_length = self.dataset_cfg.t_length * (self.dataset_cfg.skip + 1) - self.dataset_cfg.skip
        actual_overlap = actual_length - 1

        for scene_root in scene_roots:
            scene_folder = scene_root
            scene_files = sorted(list(map(str, scene_folder.glob(self.dataset_cfg.data_pattern))))
            sequence_count = len(scene_files) - actual_overlap
            scene_sequences = [scene_files[s_i: s_i + actual_length: (self.dataset_cfg.skip + 1)]
                for s_i in range(sequence_count)]
            sequences.extend(scene_sequences)

        return sequences

    def get_voxel(self, path):
        data = np.load(path)
        voxel = data['voxel_label']
        voxel = self.learning_map[voxel]
        invalid = (~data['origin_voxel_state'])
        return voxel, invalid


class Occ3DWQuery(VoxelSequenceQueryDataset):
    def __init__(self, dataset_cfg, imageset='train', force_length=-1, force_shuffle=False):
        super().__init__(dataset_cfg, imageset, force_length, force_shuffle)

        map_size = max(dataset_cfg.learning_map.keys()) + 1
        self.learning_map = np.zeros(map_size, dtype=int)
        keys = np.array(list(dataset_cfg.learning_map.keys()))
        values = np.array(list(dataset_cfg.learning_map.values()))
        self.learning_map[keys] = values

    def get_sequence(self):
        sequences = list()

        data_folder = Path(self.dataset_cfg.data_path) / self.dataset_cfg.imageset[self.imageset]
        scene_roots = list(data_folder.iterdir())

        if self.dataset_cfg.get('subset_percentage', None) is not None:
            num_of_scenes = int(len(scene_roots) * self.dataset_cfg.subset_percentage)
            scene_roots = scene_roots[:num_of_scenes]

        actual_length = self.dataset_cfg.t_length * (self.dataset_cfg.skip + 1) - self.dataset_cfg.skip
        actual_overlap = actual_length - 1

        for scene_root in scene_roots:
            scene_folder = scene_root
            scene_files = sorted(list(map(str, scene_folder.glob(self.dataset_cfg.data_pattern))))
            sequence_count = len(scene_files) - actual_overlap
            scene_sequences = [scene_files[s_i: s_i + actual_length: (self.dataset_cfg.skip + 1)]
                for s_i in range(sequence_count)]
            sequences.extend(scene_sequences)

        return sequences

    def get_voxel(self, path):
        data = np.load(path)
        voxel = data['voxel_label']
        voxel = self.learning_map[voxel]
        invalid = (~data['origin_voxel_state'])
        query, query_gt, query_int = get_query(
            voxel, self.num_classes, self.grid_size,
            self.dataset_cfg.get('query_count', 400000)
        )
        return voxel, query, query_gt, query_int, invalid


class Occ3DWHexplane(HexplaneDataset):
    def __init__(
        self,
        vae_name,
        t_length=16,
        data_path=None,
        scene_folder=None,
        angle_thr_mul=1,
        forward_thr_mul=1,
        hex_cond=True,
        cmd_cond=False,
        layout_cond=False,
        mode='train',
        voxel=False,
    ):
        super().__init__(
            vae_name, 'occ3dw', t_length, data_path, scene_folder, angle_thr_mul, forward_thr_mul,
            hex_cond, cmd_cond, layout_cond
        )

    def prepare(self):
        hexplanes, conditions, trajectories, turns = list(), list(), list(), list()
        hexplane_roots = sum([list(imageset_folder.iterdir()) for imageset_folder in self.data_folder.iterdir()], [])

        angle_thr = self.t_length * self.angle_thr_mul

        with open(Path(self.data_path).parent / 'cam_infos.pkl', 'rb') as f:
            cam_infos = pickle.load(f)
        with open(Path(self.data_path).parent / 'cam_infos_vali.pkl', 'rb') as f:
            cam_infos_vali = pickle.load(f)

        for hexplane_folder in hexplane_roots:
            scene_index = int(str(hexplane_folder.name))

            if 'training' in hexplane_folder.parts:
                info = cam_infos[scene_index]
            else:
                info = cam_infos_vali[scene_index]

            hexplane_files = sorted(list(map(str, hexplane_folder.glob('*.npy'))))

            poses = np.zeros((len(info), 16))
            for frame_index in range(len(info)):
                poses[frame_index] = info[frame_index][0]['ego2global'].reshape(-1)

            for i in range(len(hexplane_files)):
                hexplane_path = Path(hexplane_files[i])
                hexplane_number_str = hexplane_path.stem[:3]
                hexplane_number = int(hexplane_number_str)

                if self.hex_cond:
                    condition_number = hexplane_number - self.t_length
                    condition_number_str = str(condition_number).zfill(len(hexplane_number_str))
                    condition_path = hexplane_path.parent / f'{condition_number_str}_04.npy'
                    if not condition_path.exists():
                        continue
                else:
                    condition_path = hexplane_path

                if hexplane_number + self.t_length > len(poses):
                    continue

                conditions.append(condition_path)
                hexplanes.append(hexplane_path)
                trajectory = get_traj_rot(poses[hexplane_number: hexplane_number + self.t_length])
                trajectories.append(trajectory)
                angle_diff = abs(trajectory[0][2] - trajectory[-1][2])
                turn = angle_diff > angle_thr
                turns.append(turn)

        return hexplanes, conditions, trajectories, turns


class Occ3DWVoxel(VoxelDataset):
    def __init__(self, dataset_cfg, imageset='train'):
        super().__init__(dataset_cfg, imageset)

        map_size = max(dataset_cfg.learning_map.keys()) + 1
        self.learning_map = np.zeros(map_size, dtype=int)
        keys = np.array(list(dataset_cfg.learning_map.keys()))
        values = np.array(list(dataset_cfg.learning_map.values()))
        self.learning_map[keys] = values

    def get_scenes(self):
        scenes = list()

        data_folder = Path(self.dataset_cfg.data_path) / self.dataset_cfg.imageset[self.imageset]
        scene_roots = list(data_folder.iterdir())

        if self.dataset_cfg.get('subset_percentage', None) is not None:
            num_of_scenes = int(len(scene_roots) * self.dataset_cfg.subset_percentage)
            scene_roots = scene_roots[:num_of_scenes]

        for scene_root in scene_roots:
            scene_folder = scene_root
            scene_files = sorted(list(map(str, scene_folder.glob(self.dataset_cfg.data_pattern))))
            scenes.extend(scene_files)

        random.shuffle(scenes)

        return scenes

    def get_voxel(self, path):
        data = np.load(path)
        voxel = data['voxel_label']
        voxel = self.learning_map[voxel]
        invalid = (~data['origin_voxel_state'])
        return voxel, invalid
