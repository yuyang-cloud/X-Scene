from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

import occgen.utils.constants as C
from occgen.utils.dataset_utils import get_subsequence_command


class HexplaneDataset(ABC, Dataset):
    def __init__(
        self,
        vae_name,
        dataset='carlasc',
        t_length=16,
        data_path=None,
        scene_folder=None,
        angle_thr_mul=1,
        forward_thr_mul=1,
        hex_cond=True,
        cmd_cond=False,
        layout_cond=False,
    ):
        self.data_folder = Path(C.HEXPLANE_PATH) / dataset / vae_name
        self.vae_name = vae_name
        self.dataset = dataset
        self.t_length = t_length
        self.data_path = data_path
        self.scene_folder = scene_folder
        self.angle_thr_mul = angle_thr_mul
        self.forward_thr_mul = forward_thr_mul
        self.hex_cond = hex_cond
        self.cmd_cond = cmd_cond
        self.layout_cond = layout_cond
        self.hexplanes, self.conditions, self.trajectories, self.turn = self.prepare()

    def __len__(self):
        return len(self.hexplanes)

    def __getitem__(self, index):
        hexplane = np.load(self.hexplanes[index]).squeeze()

        if self.hex_cond:
            condition = np.load(self.conditions[index]).squeeze()
        else:
            condition = np.zeros_like(hexplane)

        if self.cmd_cond:
            command = get_subsequence_command(self.trajectories[index], self.angle_thr_mul, self.forward_thr_mul).value
        else:
            command = 0

        if self.layout_cond:
            layout = np.load(
                str(self.hexplanes[index])
                .replace(C.HEXPLANE_PATH, C.LAYOUT_PATH)
                .replace(self.vae_name, str(self.t_length))
            )
        else:
            layout = np.zeros(self.t_length)

        if self.voxel:
            voxel = [np.load(self.voxels[index][i]) for i in range(4)]
        else:
            voxel = None

        return {
            'x': hexplane,
            'hex_cond': condition,
            'traj_cond': self.trajectories[index],
            'path': str(self.hexplanes[index]),
            'turn': self.turn[index],
            'cmd_cond': command,
            'layout_cond': layout,
            'voxel': voxel,
        }

    @abstractmethod
    def prepare(self):
        pass
