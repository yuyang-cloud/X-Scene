import json
from pathlib import Path
import mmcv
import numpy as np
import numba as nb
from pyquaternion import Quaternion
import os
import torch

from occgen.dataset.hexplane_dataset import HexplaneDataset
from occgen.dataset.voxel_sequence_dataset import VoxelDataset, VoxelSequenceDataset
from occgen.utils.dataset_utils import get_traj_rot_from_coordinates


class Occ3DN(VoxelSequenceDataset):
    def __init__(self, dataset_cfg, imageset='train', force_length=-1, force_shuffle=False):
        with open(Path(dataset_cfg.data_path) / 'annotations.json', 'r') as f:
            self.annotations = json.load(f)
        super().__init__(dataset_cfg, imageset, force_length, force_shuffle)

    def get_sequence(self):
        sequences = list()

        scene_names = self.annotations[self.dataset_cfg.imageset[self.imageset]]

        actual_length = self.dataset_cfg.t_length * (self.dataset_cfg.skip + 1) - self.dataset_cfg.skip
        actual_overlap = actual_length - 1

        for scene_name in scene_names:
            scene = self.annotations['scene_infos'][scene_name]
            scene_files = [v['gt_path'] for v in sorted(scene.values(), key=lambda x: x['timestamp'])]
            sequence_count = len(scene_files) - actual_overlap
            scene_sequences = [scene_files[s_i: s_i + actual_length: (self.dataset_cfg.skip + 1)]
                for s_i in range(sequence_count)]
            sequences.extend(scene_sequences)

        return sequences

    def get_voxel(self, path):
        data = np.load(Path(self.dataset_cfg.data_path) / path)
        voxel = data['semantics']
        voxel = self.learning_map[voxel]
        invalid = (~data['mask_lidar'])
        return voxel, invalid


class Occ3DNHexplane(HexplaneDataset):
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
        with open(Path(data_path) / 'annotations.json', 'r') as f:
            self.annotations = json.load(f)
        self.mode = mode
        self.voxel = voxel
        super().__init__(
            vae_name, 'occ3dn', t_length, data_path, scene_folder, angle_thr_mul, forward_thr_mul,
            hex_cond, cmd_cond, layout_cond
        )

    def prepare(self):
        hexplanes, conditions, trajectories, turns = list(), list(), list(), list()
        voxels = list()

        if self.mode == 'train':
            imagesets = ['train_split']
        elif self.mode == 'valid':
            imagesets = ['val_split']
        elif self.mode == 'all':
            imagesets = ['train_split', 'val_split']
        scene_names = sum([self.annotations[imageset] for imageset in imagesets], [])

        angle_thr = self.t_length * self.angle_thr_mul

        for scene_name in scene_names:
            scene = self.annotations['scene_infos'][scene_name]
            frames = [frame for frame in sorted(scene.values(), key=lambda x: x['timestamp'])]
            coordinates = np.array([frame['ego_pose']['translation'] for frame in frames])
            hexplane_files = [frame['gt_path'].replace('.npz', '.npy') for frame in frames]
            for i in range(len(hexplane_files)):
                hexplane_path = self.data_folder / hexplane_files[i]
                hexplane_number = i

                if self.hex_cond:
                    condition_number = hexplane_number - self.t_length
                    condition_path = self.data_folder / hexplane_files[condition_number]
                    if not condition_path.exists():
                        continue
                else:
                    condition_path = hexplane_path

                if not hexplane_path.exists():
                    continue

                conditions.append(condition_path)
                hexplanes.append(hexplane_path)
                trajectory = get_traj_rot_from_coordinates(
                    coordinates[hexplane_number: hexplane_number + self.t_length]
                )
                trajectories.append(trajectory)
                angle_diff = abs(trajectory[0][2] - trajectory[-1][2])
                turn = angle_diff > angle_thr
                turns.append(turn)
                voxels.append([
                    Path('occ3dn') / hexplane_files[j].replace('.npy', '.npz')
                for j in range(i, i+4)])

        self.voxels = voxels
        return hexplanes, conditions, trajectories, turns


class Occ3DNVoxel(VoxelDataset):
    def __init__(self, dataset_cfg, imageset='train'):
        self.occ_root = os.path.join(dataset_cfg.data_path, 'gts')
        self.data_infos = mmcv.load(os.path.join(dataset_cfg.info_path, f'nuscenes_infos_{imageset}.pkl'))['infos']

        self.grid_size = np.array(dataset_cfg.grid_size).astype(np.int64)
        self.pc_range = np.array(dataset_cfg.pc_range)
        self.voxel_size = np.array(dataset_cfg.voxel_size)

        super().__init__(dataset_cfg, imageset)

        self.tokens = [Path(scene).parent.name for scene in self.scenes]

    def get_scenes(self):
        scenes = []
        lidar2ego_list = []
        for data in sorted(self.data_infos, key=lambda x: x['token']):
            gt_occ_path = os.path.join(self.occ_root, data['scene_name'], data['token'], 'labels.npz')
            scenes.append(gt_occ_path)
            lidar2ego = np.eye(4).astype(np.float64)
            lidar2ego[:3, :3] = Quaternion(
                data["lidar2ego_rotation"]).rotation_matrix
            lidar2ego[:3, 3] = data["lidar2ego_translation"]
            lidar2ego_list.append(lidar2ego)

        return scenes, lidar2ego_list

    def get_voxel(self, path, lidar2ego):
        data = np.load(path)
        voxel = data['semantics']
        voxel = self.learning_map[voxel]
        invalid = (~data['mask_lidar'])
        return voxel, invalid
    
    def trans_occ_ego2lidar(self, voxel_label, lidar2ego):
        voxel_label[voxel_label == 17] = 0
        xyzl = []
        for i in range(18):
            xyz = torch.nonzero(torch.Tensor(voxel_label) == i, as_tuple=False)
            xyzlabel = torch.nn.functional.pad(xyz, (0,1),'constant', value=i)
            xyzl.append(xyzlabel)
        xyzl = torch.cat(xyzl, dim=0)
        xyz = xyzl[:, :-1].numpy()
        label = xyzl[:, -1].numpy()

        invalid = (label == 0).astype(np.uint8)
        label[invalid == 1] = 255

        # transfer the ego_coord to lidar_coord
        xyz_ego = self.voxel2world(xyz)
        xyz_lidar, pc_range_lidar = self.ego2lidar_transform(xyz_ego, lidar2ego)
        pcd_np_cor = self.world2voxel_with_pcrange(xyz_lidar, pc_range_lidar)
            
        # make sure the point is in the grid
        pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), self.grid_size - 1)
        pcd_np = np.concatenate([pcd_np_cor, label[:, None]], axis=-1)

        # 255: noise, 1-16 normal classes, 0 unoccupied
        pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
        pcd_np = pcd_np.astype(np.int64)
        occupied = (pcd_np[:, -1] != 0)
        pcd_np = pcd_np[occupied]

        processed_label = np.ones(self.grid_size.astype(np.uint8), dtype=np.uint8) * 0
        processed_label = nb_process_label(processed_label, pcd_np)
        processed_label[processed_label == 255] = 0
        return processed_label
    
    def voxel2world(self, voxel):
        """
        voxel: [N, 3]
        """
        return voxel * self.voxel_size[None, :] + self.pc_range[:3][None, :]

    def world2voxel_with_pcrange(self, world, pc_range):
        """
        world: [N, 3]
        """
        return (world - pc_range[:3][None, :]) / self.voxel_size[None, :]

    def ego2lidar_transform(self, xyz_ego, lidar2ego):
        """
        xyz_ego: [N, 3]
        lidar2ego: [4, 4]
        """
        ego2lidar = np.linalg.inv(lidar2ego)
        xyz_ego = np.concatenate([xyz_ego, np.ones((len(xyz_ego), 1))], axis=-1)
        pc_range_ego = np.concatenate([self.pc_range[:3][None], self.pc_range[3:][None]], axis=0)
        pc_range_ego = np.concatenate([pc_range_ego, np.ones((len(pc_range_ego), 1))], axis=-1)
        xyz_lidar = (xyz_ego @ ego2lidar.T)[:, :3]
        pc_range_lidar = (pc_range_ego @ ego2lidar.T)[:, :3]
        pc_range_lidar = np.concatenate([pc_range_lidar.min(0), pc_range_lidar.max(0)], axis=0)
        return xyz_lidar, pc_range_lidar

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)

    return processed_label
