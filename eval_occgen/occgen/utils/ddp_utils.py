import logging
from pathlib import Path

import torch.distributed as dist
import wandb


def func_rank_0(func):
    def wrapper(*args, **kwargs):
        if dist.get_rank() != 0:
            return None
        return func(*args, **kwargs)

    return wrapper


def rank_0():
    return dist.get_rank() == 0


def create_logger(log_path):
    if dist.get_rank() == 0:
        log_path = Path(log_path) / 'log.txt'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(str(log_path))]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


@func_rank_0
def write_text(text):
    print(text, end='')


@func_rank_0
def flush_text():
    print('\r', end='')


@func_rank_0
def print_text(*args, **kwargs):
    print(*args, **kwargs)


@func_rank_0
def wandb_log(*args, **kwargs):
    wandb.log(*args, **kwargs)
