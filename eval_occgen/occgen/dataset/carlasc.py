from pathlib import Path

import numpy as np

from occgen.dataset.hexplane_dataset import HexplaneDataset
from occgen.dataset.voxel_sequence_dataset import VoxelDataset, VoxelSequenceDataset
from occgen.utils.dataset_utils import get_traj_rot


class CarlaSC(VoxelSequenceDataset):
    def __init__(self, dataset_cfg, imageset='train', force_length=-1, force_shuffle=False):
        super().__init__(dataset_cfg, imageset, force_length, force_shuffle)

    def get_sequence(self):
        sequences = list()

        data_folder = Path(self.dataset_cfg.data_path) / self.dataset_cfg.imageset[self.imageset]
        scene_roots = list(data_folder.iterdir())

        actual_length = self.dataset_cfg.t_length * (self.dataset_cfg.skip + 1) - self.dataset_cfg.skip
        actual_overlap = actual_length - 1

        for scene_root in scene_roots:
            scene_folder = scene_root / self.dataset_cfg.scene_folder
            scene_files = sorted(list(map(str, scene_folder.glob(self.dataset_cfg.data_pattern))))
            sequence_count = len(scene_files) - actual_overlap
            scene_sequences = list()
            for s_i in range(sequence_count):
                scene_sequences.append(scene_files[s_i: s_i + actual_length: (self.dataset_cfg.skip + 1)])
            sequences.extend(scene_sequences)

        return sequences

    def get_voxel(self, path):
        voxel = np.fromfile(path, dtype=np.uint32).reshape(self.grid_size)
        voxel = self.learning_map[voxel]
        valid = np.fromfile(path.replace('label', 'bin'), dtype=np.float32).reshape(self.grid_size)
        invalid = np.zeros(valid.shape, dtype=np.uint8)
        invalid[valid == 0] = 1
        return voxel, invalid


class CarlaSCHexplane(HexplaneDataset):
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
            vae_name, 'carlasc', t_length, data_path, scene_folder, angle_thr_mul, forward_thr_mul,
            hex_cond, cmd_cond, layout_cond
        )

    def prepare(self):
        hexplanes, conditions, trajectories, turns = list(), list(), list(), list()
        hexplane_roots = sum([list(imageset_folder.iterdir()) for imageset_folder in self.data_folder.iterdir()], [])

        angle_thr = self.t_length * self.angle_thr_mul

        for hexplane_folder in hexplane_roots:
            hexplane_files = sorted(list(map(str, hexplane_folder.glob('*.npy'))))
            poses = np.loadtxt(
                Path(self.data_path) / Path(*hexplane_folder.parts[-2:]) /
                self.scene_folder.split('/')[0] / 'poses.txt'
            )
            for i in range(len(hexplane_files)):
                hexplane_path = Path(hexplane_files[i])
                hexplane_number_str = hexplane_path.stem
                hexplane_number = int(hexplane_number_str)

                if self.hex_cond:
                    condition_number = hexplane_number - self.t_length
                    condition_number_str = str(condition_number).zfill(len(hexplane_number_str))
                    # condition_path = (Path('hexplane/carlasc/single_pcd') / hexplane_path.parent.parent.name /
                    #                   hexplane_path.parent.name / f'{condition_number_str}.npy')
                    condition_path = hexplane_path.parent / f'{condition_number_str}.npy'
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


class CarlaSCVoxel(VoxelDataset):
    def __init__(self, dataset_cfg, imageset='train'):
        super().__init__(dataset_cfg, imageset)

    def get_scenes(self):
        scenes = list()

        data_folder = Path(self.dataset_cfg.data_path) / self.dataset_cfg.imageset[self.imageset]
        scene_roots = list(data_folder.iterdir())

        for scene_root in scene_roots:
            scene_folder = scene_root / self.dataset_cfg.scene_folder
            scene_files = sorted(list(map(str, scene_folder.glob(self.dataset_cfg.data_pattern))))
            scenes.extend(scene_files)

        return scenes

    def get_voxel(self, path):
        voxel = np.fromfile(path, dtype=np.uint32).reshape(self.grid_size)
        voxel = self.learning_map[voxel]
        valid = np.fromfile(path.replace('label', 'bin'), dtype=np.float32).reshape(self.grid_size)
        invalid = np.zeros(valid.shape, dtype=np.uint8)
        invalid[valid == 0] = 1
        return voxel, invalid


if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('conf/vae/dataset/carlasc.yaml')
    dataset = CarlaSC(cfg, 'valid')
    # dataset = CarlaSCHexplane('1801_c16', 16, 'carlasc/Cartesian', 'cartesian/evaluation')
    # dataset = CarlaSCMixed(cfg, 'train', '1005_c_t8_15', '0915_hex_traj')
    print(len(dataset))
    from tqdm import tqdm
    for i in tqdm(range(len(dataset))):
        dataset[i]
    # cfg = OmegaConf.load('conf/vae/dataset/carlasc.yaml')
    # cfg.t_length = 16
    # dataset = CarlaSC(cfg, shuffle=True)

    # out_dir = Path('out/voxel/carla_aug')
    # out_dir.mkdir(parents=True, exist_ok=True)

    # for sample_idx in range(8):
    #     for aug_type in range(8):
    #         aug_sample = dataset.get_item_with_aug(sample_idx, aug_type)
    #         aug_voxels = aug_sample[0]

    #         aug_dir = out_dir / f"{sample_idx}_{aug_type}"
    #         aug_dir.mkdir(exist_ok=True)

    #         for t in range(aug_voxels.shape[0]):
    #             voxel = aug_voxels[t].numpy().astype(np.uint8)
    #             filename = aug_dir / f"{t}.label"
    #             voxel.tofile(str(filename))

    # print(f"Augmented voxels saved to {out_dir}")
