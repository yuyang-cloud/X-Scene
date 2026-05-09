from pathlib import Path

import numpy as np

from occgen.dataset.voxel_sequence_dataset import VoxelDataset


class SemKittiVoxel(VoxelDataset):
    def __init__(self, dataset_cfg, imageset='train'):
        super().__init__(dataset_cfg, imageset)

        learning_map_dict = dataset_cfg.learning_map
        max_key = max(learning_map_dict.keys())
        self.learning_map = np.zeros(max_key + 1, dtype=np.int64)
        for k, v in learning_map_dict.items():
            self.learning_map[k] = v

    def get_scenes(self):
        data_folder = Path(self.dataset_cfg.data_path)
        folders = self.dataset_cfg.imageset[self.imageset].split(',')
        scenes = list()
        for folder in folders:
            scenes.extend(
                list(
                    data_folder.glob(
                        f'{folder}/{self.dataset_cfg.scene_folder}/{self.dataset_cfg.data_pattern}'
                    )
                )
            )
        return scenes

    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1
        return uncompressed

    def get_voxel(self, path):
        voxel = np.fromfile(path, dtype=np.uint16).reshape(self.grid_size)
        voxel = self.learning_map[voxel]
        invalid = np.fromfile(str(path).replace('label', 'invalid'), dtype=np.uint8)
        invalid = self.unpack(invalid).astype(np.float32).reshape(self.grid_size)
        return voxel, invalid
