from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class F3DDataset(Dataset):
    def __init__(self, root_dir, hwd, learningmap=None):
        self.root_dir = Path(root_dir)
        self.h, self.w, self.d = hwd
        if hasattr(learningmap, 'detach'):
            learningmap = learningmap.detach().cpu().numpy()
        self.learningmap = learningmap

        self.files = sorted(self.root_dir.glob("*.npz"), key=lambda x: x.name)
        if len(self.files) == 0:
            self.files = sorted(self.root_dir.glob("*.label"), key=lambda x: x.name)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz or .label files were found in {self.root_dir}.")
        self.tokens = [self._get_token(path) for path in self.files]

    @staticmethod
    def _get_token(path):
        return path.stem

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        if file_path.suffix == '.npz':
            data = np.load(file_path)
            if 'occ' in data.files:
                data = data['occ']
            elif 'arr_0' in data.files:
                data = data['arr_0']
            else:
                raise KeyError(f"Unsupported npz keys in {file_path}: {data.files}")
            data = data.reshape(self.h, self.w, self.d).astype(np.int32)
        else:
            data = np.fromfile(file_path, dtype=np.uint8).reshape(self.h, self.w, self.d).astype(np.int32)

        if self.learningmap is not None:
            data = self.learningmap[data]
        return {
            'voxel': data,
            'path': self.files[idx].as_posix(),
        }
