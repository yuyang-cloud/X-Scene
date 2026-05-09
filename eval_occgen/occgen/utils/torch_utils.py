import random

import numpy as np
import torch
import torch.distributed as dist


def setup_ddp(verbose=True):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    if verbose:
        print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")
    return rank, device


def cleanup_ddp():
    dist.destroy_process_group()


def set_seed(global_seed, deterministic=False, rank=0):
    if global_seed is not None:
        seed = global_seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_tf32(use_tf32=True):
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
