import warnings
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from tqdm import tqdm

import occgen.utils.constants as C
from occgen.dataset.f2d import F2DDataset
from occgen.utils.ddp_utils import func_rank_0, print_text, rank_0, wandb_log
from occgen.utils.file_utils import get_latest_ckpt, save_cfg
from occgen.utils.torch_utils import set_seed, set_tf32, setup_ddp
from occgen.vae.f2d_ae import F2D

warnings.filterwarnings("ignore")


class F2DTrainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.amp = cfg.trainer.amp
        self.log_freq = cfg.trainer.log_interval
        self.num_epochs = cfg.trainer.num_epochs

        self.train_loader, self.valid_loader = self.setup_dataloaders()

        self.model = F2D()
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.device])

        self.optimizer, self.scheduler = self.setup_optimizer()
        if self.amp:
            self.scaler = GradScaler()

        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')

        self.load_checkpoint(self.get_resume_ckpt())

    def setup_dataloaders(self):
        trainset_path = self.cfg.dataset.get('trainset_path')
        validset_path = self.cfg.dataset.get('validset_path')
        if trainset_path is not None and validset_path is not None:
            train_dataset = F2DDataset(trainset_path)
            valid_dataset = F2DDataset(validset_path)
        elif self.cfg.dataset.get('path') is not None:
            dataset = F2DDataset(self.cfg.dataset.path)
            generator = torch.Generator().manual_seed(self.cfg.trainer.seed or 0)
            valid_size = max(1, int(len(dataset) * 0.1))
            train_size = len(dataset) - valid_size
            train_dataset, valid_dataset = torch.utils.data.random_split(
                dataset, [train_size, valid_size], generator=generator
            )
        else:
            raise ValueError("F2D training requires dataset.trainset_path/validset_path or dataset.path.")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg.dataset.batch_size, sampler=train_sampler,
            num_workers=self.cfg.dataset.num_workers, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=self.cfg.dataset.batch_size, sampler=valid_sampler,
            num_workers=self.cfg.dataset.num_workers, pin_memory=True)
        return train_loader, valid_loader

    def setup_optimizer(self):
        optim_cls = dict(adam=Adam, adamw=AdamW)[self.cfg.model.optimizer_type]
        opt = optim_cls(
            self.model.parameters(), lr=self.cfg.model.learning_rate, weight_decay=self.cfg.model.weight_decay
        )
        sch = None
        return opt, sch

    def get_resume_ckpt(self):
        ckpt_path = None
        if self.cfg.trainer.auto_resume:
            ckpt_path = get_latest_ckpt(C.CKPT_PATH, self.cfg.name)
        if self.cfg.trainer.resume_ckpt is not None:
            ckpt_path = self.cfg.trainer.resume_ckpt
        return ckpt_path

    @func_rank_0
    def save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'global_step': self.global_step,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        if checkpoint_path is None:
            return
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{self.device}')
        self.current_epoch = checkpoint['epoch'] + 1
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.global_step = checkpoint['global_step']
        print_text(f'Loaded checkpoint {checkpoint_path}')

    def train_epoch(self):
        self.model.train()
        self.train_loader.sampler.set_epoch(self.current_epoch)
        epoch_loss = 0
        pbar = tqdm(
            self.train_loader, desc=f"Train {self.current_epoch} / {self.num_epochs}", disable=not rank_0(), leave=False
        )
        for batch in pbar:
            imgs = batch.to(self.device)
            self.optimizer.zero_grad()
            with autocast(enabled=self.amp):
                outputs = self.model.module.train_forward(imgs)
                gt = F.interpolate(imgs, size=(192, 192), mode='bilinear', align_corners=False)
                loss = F.mse_loss(outputs, gt)
            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            epoch_loss += loss.item()
            self.global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        return epoch_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            pbar = tqdm(
                self.valid_loader, desc=f"Val {self.current_epoch} / {self.num_epochs}", disable=not rank_0(),
                leave=False
            )
            for batch in pbar:
                imgs = batch.to(self.device)
                outputs = self.model.module.train_forward(imgs)
                gt = F.interpolate(imgs, size=(192, 192), mode='bilinear', align_corners=False)
                loss = F.mse_loss(outputs, gt)
                epoch_loss += loss.item()
        return epoch_loss / len(self.valid_loader)

    def fit(self):
        if rank_0():
            wandb.init(
                name=str(self.cfg.name), dir=C.WANDB_PATH, project=C.WANDB_PROJECT,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                mode="offline" if self.cfg.trainer.debug else "online"
            )
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.num_epochs):
            if epoch == 0:
                self.model.module.net.freeze_encoder()
            else:
                self.model.module.net.unfreeze_encoder()
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            train_loss_tensor = torch.tensor(train_loss, device=self.device)
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = (train_loss_tensor / dist.get_world_size()).item()
            val_loss = (val_loss_tensor / dist.get_world_size()).item()

            if rank_0():
                wandb_log({'train_loss': train_loss, 'val_loss': val_loss}, step=self.global_step)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(Path(C.CKPT_PATH) / str(self.cfg.name) / C.CKPT_LAST)

        if rank_0():
            wandb.finish()


@hydra.main(config_path=C.TRAIN_HYDRA_ROOT, config_name=C.F2D_TRAIN_DEFAULT, version_base=None)
def main(cfg: DictConfig) -> None:
    set_tf32(cfg.trainer.tf32)
    rank, device = setup_ddp()
    set_seed(cfg.trainer.seed, cfg.trainer.deterministic, rank)
    print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")
    if rank_0():
        save_cfg(cfg, print_cfg=False)
    trainer = F2DTrainer(cfg, device)
    trainer.fit()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
