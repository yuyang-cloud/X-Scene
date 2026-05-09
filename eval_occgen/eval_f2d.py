import argparse
from pathlib import Path

import torch
import torch_fidelity
from omegaconf import OmegaConf
from torch_fidelity import register_feature_extractor

import occgen.utils.constants as C
from occgen.dataset.f2d import F2DDataset
from occgen.utils.file_utils import get_latest_ckpt, get_vae_ckpt, load_cached_conf
from occgen.vae.f2d_ae import F2D


class TensorDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, max_cnt=None):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        if max_cnt is not None:
            self.indices = self.indices[:max_cnt]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]['img']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='F2D')
    parser.add_argument('-f', '--folder', type=str, default='occ3dn')
    parser.add_argument('--gt-path', type=str, default=None)
    parser.add_argument('--sample-path', type=str, default=None)
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--epoch', type=int, default=None)
    args = parser.parse_args()
    return args


def resolve_path(path, default_root=None):
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute() or resolved.exists() or resolved.parent != Path('.') or default_root is None:
        return resolved
    return Path(default_root) / resolved


def validate_matching_tokens(gt_tokens, sample_tokens):
    if gt_tokens == sample_tokens:
        return
    if len(gt_tokens) != len(sample_tokens):
        raise ValueError(
            f"Token count mismatch: ground truth has {len(gt_tokens)} files, "
            f"samples have {len(sample_tokens)} files."
        )
    for idx, (gt_token, sample_token) in enumerate(zip(gt_tokens, sample_tokens)):
        if gt_token != sample_token:
            raise ValueError(
                f"Token mismatch at index {idx}: ground truth '{gt_token}' != sample '{sample_token}'."
            )
    raise ValueError("Tokens do not match between datasets.")


def select_checkpoint(args, cfg):
    if args.epoch is not None:
        raise ValueError("F2D training saves the best-loss checkpoint as last.ckpt; --epoch is not supported.")
    ckpt_path = get_latest_ckpt(C.CKPT_PATH, cfg.name)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for run '{cfg.name}' in {C.CKPT_PATH}.")
    return ckpt_path


def main():
    register_feature_extractor('f2d', F2D)

    args = get_args()
    cfg = load_cached_conf(args.name)

    OmegaConf.set_struct(cfg, False)
    ckpt_path = select_checkpoint(args, cfg)

    gt_path = resolve_path(args.gt_path or cfg.dataset.get('eval_gt_img_path') or cfg.dataset.validset_path)
    sample_path = resolve_path(
        args.sample_path or cfg.dataset.get('eval_gen_img_path') or args.folder,
        Path(C.OUT_PATH) / C.GEN_IMG_PATH
    )

    train_dataset = F2DDataset(gt_path, return_dict=True)
    sample_dataset = F2DDataset(sample_path, return_dict=True)
    train_dataset_warpper = TensorDatasetWrapper(train_dataset, max_cnt=10000)
    sample_dataset_warpper = TensorDatasetWrapper(sample_dataset, max_cnt=10000)

    validate_matching_tokens(train_dataset.tokens, sample_dataset.tokens)

    torch_fidelity.calculate_metrics(
        input1=train_dataset_warpper,
        input2=sample_dataset_warpper,
        batch_size=64,
        isc=True,
        fid=True,
        kid=True,
        prc=True,
        cuda=True,
        verbose=True,
        feature_extractor='f2d',  # use pre-trained VGG16 to calculate P,R
        feature_extractor_weights_path=ckpt_path
        # feature_extractor=None, # use default InceptionV3 to calculate ISC, FID, KID
    )
    print()


if __name__ == '__main__':
    main()
