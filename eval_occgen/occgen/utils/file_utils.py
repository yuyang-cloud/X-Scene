from pathlib import Path

from omegaconf import OmegaConf

import occgen.utils.constants as C


def get_ckpts_sorted(root, pattern, func=None):
    matches = list(Path(root).glob(pattern))
    if len(matches) == 0:
        return [None]
    if func is not None:
        matches.sort(key=func, reverse=True)
    return matches


def get_vae_ckpt(root, prefix, epoch='*'):
    return get_ckpts_sorted(root, f'{prefix}*/{epoch}_mIoU_*.ckpt', lambda x: float(x.stem.split('_')[2]))


def get_latest_ckpt(root, prefix):
    matches = get_ckpts_sorted(root, f'{prefix}*/{C.CKPT_LAST}')
    assert len(matches) == 1
    return matches[0]


def get_dit_ckpt(root, prefix, step=-1):
    return get_ckpts_sorted(root, f'{prefix}*/*{"" if step == -1 else step}.ckpt', lambda x: int(x.stem))


def load_cached_conf(vae_name):
    cfg_candidates = [
        Path(C.CFG_PATH) / vae_name / C.CFG_FILENAME,
        Path(C.CKPT_PATH) / vae_name / C.CFG_FILENAME,
    ]
    for cfg_path in cfg_candidates:
        if cfg_path.exists():
            return OmegaConf.load(cfg_path)
    candidates = ', '.join(str(path) for path in cfg_candidates)
    raise FileNotFoundError(f"Config file not found. Checked: {candidates}")


def save_cfg(cfg, print_cfg=True):
    config_yaml = OmegaConf.to_yaml(cfg)
    cfg_path = Path(C.CFG_PATH) / str(cfg.name) / C.CFG_FILENAME
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, 'w') as f:
        f.write(config_yaml)
    if print_cfg:
        print(config_yaml)
