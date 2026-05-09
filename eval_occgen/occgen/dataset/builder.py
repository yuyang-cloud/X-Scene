import torch
from torch.utils.data import DataLoader, DistributedSampler

from occgen.dataset.carlasc import CarlaSC, CarlaSCHexplane, CarlaSCVoxel
from occgen.dataset.occ3dn import Occ3DN, Occ3DNHexplane, Occ3DNVoxel
from occgen.dataset.occ3dw import Occ3DW, Occ3DWHexplane, Occ3DWQuery, Occ3DWVoxel
from occgen.dataset.semkitti import SemKittiVoxel
from occgen.utils.dit_train_utils import DistributedTurningSampler
from occgen.utils.torch_utils import seed_worker


def get_voxel_seq_dataloaders(cfg, shuffle=False):
    if cfg.model.get('decoder_type', '') == 'query':
        dataset_class = {
            'occ3dw': Occ3DWQuery,
        }[cfg.dataset.dataset]
    else:
        dataset_class = {
            'carlasc': CarlaSC,
            'occ3dw': Occ3DW,
            'occ3dn': Occ3DN,
        }[cfg.dataset.dataset]

    train_dataset = dataset_class(
        cfg.dataset,
        imageset='train',
        force_length=cfg.trainer.debug_length,
        force_shuffle=shuffle,
    )
    valid_dataset = dataset_class(
        cfg.dataset,
        imageset='valid',
        force_length=cfg.trainer.debug_length,
        force_shuffle=shuffle,
    )

    seed = cfg.trainer.seed or 0
    train_sampler = DistributedSampler(train_dataset, seed=seed)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        sampler=train_sampler,
        num_workers=cfg.dataset.num_workers,
        worker_init_fn=seed_worker,
        generator=train_generator,
        pin_memory=True
    )

    valid_sampler = DistributedSampler(valid_dataset, seed=seed)
    valid_generator = torch.Generator()
    valid_generator.manual_seed(seed)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        sampler=valid_sampler,
        num_workers=cfg.dataset.num_workers,
        worker_init_fn=seed_worker,
        generator=valid_generator,
        pin_memory=True
    )
    return train_dataloader, valid_dataloader


def get_hexplane_dataloaders(vae_cfg, cfg):
    dataset_class = {
        'carlasc': CarlaSCHexplane,
        'occ3dn': Occ3DNHexplane,
        'occ3dw': Occ3DWHexplane,
    }[vae_cfg.dataset.dataset]
    dataset = dataset_class(
        vae_cfg.name,
        vae_cfg.dataset.t_length,
        vae_cfg.dataset.data_path,
        vae_cfg.dataset.scene_folder,
        cfg.dataset.angle_thr_mul,
        cfg.dataset.get('forward_thr_mul', 1),
        cfg.model.get('hex_cond', False),
        cfg.model.get('cmd_cond', False),
        cfg.model.get('layout_cond', False),
        cfg.model.get('mode', 'train'),
        cfg.model.get('voxel', False),
    )

    seed = cfg.trainer.seed or 0
    sampler = DistributedTurningSampler(
        dataset,
        repeat_factor=cfg.dataset.turn_repeat_factor,
        seed=seed
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        sampler=sampler,
        num_workers=cfg.dataset.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
        drop_last=True
    )
    return dataloader


def get_voxel_dataloaders(cfg):
    dataset_class = {
        'carlasc': CarlaSCVoxel,
        'occ3dn': Occ3DNVoxel,
        'occ3dn_nomap': Occ3DNVoxel,
        'occ3dw': Occ3DWVoxel,
        'semkitti': SemKittiVoxel,
    }[cfg.dataset.dataset]

    train_dataset = dataset_class(
        cfg.dataset,
        imageset='train',
    )
    valid_imageset = cfg.dataset.get('valid_imageset')
    if valid_imageset is None:
        valid_imageset = 'val' if cfg.dataset.dataset in ('occ3dn', 'occ3dn_nomap') else 'valid'
    valid_dataset = dataset_class(
        cfg.dataset,
        imageset=valid_imageset,
    )

    seed = cfg.trainer.seed or 0
    train_sampler = DistributedSampler(train_dataset, seed=seed)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        sampler=train_sampler,
        num_workers=cfg.dataset.num_workers,
        worker_init_fn=seed_worker,
        generator=train_generator,
        pin_memory=True
    )

    valid_sampler = DistributedSampler(valid_dataset, seed=seed)
    valid_generator = torch.Generator()
    valid_generator.manual_seed(seed)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        sampler=valid_sampler,
        num_workers=cfg.dataset.num_workers,
        worker_init_fn=seed_worker,
        generator=valid_generator,
        pin_memory=True
    )
    return train_dataloader, valid_dataloader
