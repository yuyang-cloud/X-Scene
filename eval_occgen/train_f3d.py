import time
import warnings
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from tqdm import tqdm

import occgen.utils.constants as C
from occgen.dataset.builder import get_voxel_dataloaders
from occgen.utils.ddp_utils import func_rank_0, print_text, rank_0, wandb_log
from occgen.utils.file_utils import get_latest_ckpt, save_cfg
from occgen.utils.loss_utils import build_losses
from occgen.utils.metrics import Metrics
from occgen.utils.torch_utils import set_seed, set_tf32, setup_ddp
from occgen.utils.vae_train_utils import get_pred_label
from occgen.vae.f3d_ae import F3D

warnings.filterwarnings("ignore")


class F3DTrainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.amp = cfg.trainer.amp
        self.h, self.w, self.d = cfg.dataset.grid_size
        self.num_classes = cfg.dataset.num_classes
        self.log_freq = cfg.trainer.log_interval
        self.num_epochs = cfg.trainer.num_epochs

        self.train_loader, self.valid_loader = get_voxel_dataloaders(cfg)

        self.model = F3D(num_classes=self.num_classes)

        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.device])

        self.loss_history = dict()
        self.loss_names = list(cfg.model.loss_weights.keys())
        self.loss_weights = dict(**cfg.model.loss_weights)
        self.loss_functions = build_losses(cfg.model, cfg=cfg, device=self.device)

        self.optimizer, self.scheduler = self.setup_optimizer()
        if self.amp:
            self.scaler = GradScaler()

        self.metrics = self.setup_metrics()

        self.global_step = 0
        self.current_epoch = 0
        self.best_miou = 0.0

        self.load_checkpoint(self.get_resume_ckpt())

    def setup_metrics(self):
        metric_names = ['valid']
        return {name: Metrics(self.num_classes, self.device) for name in metric_names}

    def setup_optimizer(self):
        optim_cls = dict(
            adam=Adam,
            adamw=AdamW,
        )[self.cfg.model.optimizer_type]
        opt = optim_cls(
            self.model.parameters(),
            lr=self.cfg.model.learning_rate, weight_decay=self.cfg.model.weight_decay
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
            'best_miou': self.best_miou,
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
        self.best_miou = checkpoint['best_miou']
        self.global_step = checkpoint['global_step']

        print_text(f'Loaded checkpoint {checkpoint_path}')

    def add_to_metrics(self, pred, gt, invalid, name='train'):
        assert name != 'train'
        metrics = self.metrics[name]
        pred = get_pred_label(pred.dense()[..., :invalid.shape[-3], :invalid.shape[-2], :invalid.shape[-1], :], dim=-1)
        masks = invalid != 1
        eval_output = pred[masks]
        eval_label = gt[masks]
        return metrics.add_batch(eval_output.to(torch.int32), eval_label.to(torch.int32))

    def log_metrics_and_reset(self, name='train'):
        metrics = self.metrics[name]
        iou, miou_all, miou_sem, _ = metrics.get_metrics_dist()
        metrics.reset()

        wandb_log({f'{name}/mIoU_sem': miou_sem}, step=self.global_step)

    def log_loss(self, name='train'):
        if name in self.loss_history:
            losses = self.loss_history[name]
            length = len(self.train_loader) if name == 'train' else len(self.valid_loader)
            for loss_name in losses.keys():
                wandb_log({f'{name}/{loss_name}': losses[loss_name] / length}, step=self.global_step)

    def compute_losses(self, pred, gt, model_out):
        losses = dict()
        for loss_name in self.loss_names:
            loss_func = self.loss_functions[loss_name]
            if loss_name == 'ce':
                loss_params = pred, gt
            elif loss_name == 'lovasz':
                loss_params = pred.reshape(-1, self.num_classes), gt.reshape(-1, )
            else:
                raise ValueError(f"Unsupported F3D loss: {loss_name}")
            losses[loss_name] = loss_func(*loss_params)
        return losses

    def step(self, vox, *args, **kwargs):
        model_out = self.model.module.train_forward(vox, *args, **kwargs)

        pred = model_out['pred'].F
        coords = model_out['pred'].C
        gt = vox[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]
        losses = self.compute_losses(pred, gt, model_out)
        losses['loss'] = sum([self.loss_weights[loss_name] * losses[loss_name] for loss_name in losses.keys()])

        return {
            'losses': losses,
            'out': model_out['pred'],
        }

    def forward_step(self, batch, name='train'):
        vox, invalid = batch['voxel'], batch['invalid']
        vox, invalid = vox.to(self.device), invalid.to(self.device)

        kwargs = dict()

        if self.amp:
            with autocast():
                step_out = self.step(vox, **kwargs)
        else:
            step_out = self.step(vox, **kwargs)

        losses = step_out['losses']
        for loss_name in losses.keys():
            self.loss_history[name][loss_name] = self.loss_history[name][loss_name] + (losses[loss_name].item())

        if name == 'train' and self.global_step % self.log_freq == 0:
            wandb_log({'step/loss': losses['loss'].item()}, step=self.global_step)
            wandb_log({'step/cuda': torch.cuda.memory_reserved() / 2 ** 30}, step=self.global_step)

        if name == 'valid':
            pred = step_out['out']
            _, miou_all, miou_sem, _ = self.add_to_metrics(pred, vox, invalid, name=name)

        return step_out

    def backward_step(self, loss):
        self.optimizer.zero_grad()
        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.global_step >= self.cfg.model.grad_clip_step and self.cfg.model.grad_max_norm > 0:
            if self.amp:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.model.grad_max_norm)
            if self.global_step % self.log_freq == 0:
                wandb_log({'grad_norm': grad_norm}, step=self.global_step)

        if self.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def train_val_epoch_end(self, name='train'):
        self.log_loss(name)
        if name == 'valid':
            self.log_metrics_and_reset(name)

    def train_epoch(self):
        self.model.train()
        self.loss_history['train'] = {loss_name: 0. for loss_name in self.loss_names + ['loss']}

        self.train_loader.sampler.set_epoch(self.current_epoch)
        epoch_start_time = time.time()

        pbar = tqdm(
            self.train_loader, desc=f"Train {self.current_epoch} / {self.num_epochs}",
            disable=not rank_0(), leave=False
        )
        for batch in pbar:
            step_out = self.forward_step(batch, name='train')
            loss = step_out['losses']['loss']

            self.backward_step(loss)

            lr = self.optimizer.param_groups[0]['lr']
            if self.global_step % self.log_freq == 0:
                wandb_log({'lr': lr}, step=self.global_step)

            self.global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        epoch_end_time = time.time()
        wandb_log({'epoch_duration': epoch_end_time - epoch_start_time}, step=self.global_step)
        wandb_log({'epoch': self.current_epoch}, step=self.global_step)

        self.train_val_epoch_end()

    def validate_epoch(self, training=False):
        self.model.eval()
        self.loss_history['valid'] = {loss_name: 0. for loss_name in self.loss_names + ['loss']}

        with torch.no_grad():
            pbar = tqdm(
                self.valid_loader, desc=f"Val {self.current_epoch} / {self.num_epochs}",
                disable=not rank_0(), leave=False
            )
            for batch in pbar:
                self.forward_step(batch, name='valid')

        if training:
            val_miou = self.metrics['valid'].get_metrics_dist()[2]
            if self.best_miou <= val_miou:
                self.save_checkpoint(
                    Path(C.CKPT_PATH) / str(self.cfg.name) /
                    C.CKPT_FILENAME_RULE.format(self.current_epoch, val_miou)
                )
            self.best_miou = max(val_miou, self.best_miou)
            self.train_val_epoch_end(name='valid')

    def fit(self):
        if rank_0():
            wandb.init(
                name=str(self.cfg.name),
                dir=C.WANDB_PATH,
                project=C.WANDB_PROJECT,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                mode="offline" if self.cfg.trainer.debug else "online"
            )

        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.train_epoch()
            self.validate_epoch(training=True)
            self.save_checkpoint(Path(C.CKPT_PATH) / str(self.cfg.name) / C.CKPT_LAST)

        if rank_0():
            wandb.finish()


@hydra.main(config_path=C.TRAIN_HYDRA_ROOT, config_name=C.F3D_TRAIN_DEFAULT, version_base=None)
def main(cfg: DictConfig) -> None:
    set_tf32(cfg.trainer.tf32)
    rank, device = setup_ddp()
    set_seed(cfg.trainer.seed, cfg.trainer.deterministic, rank)
    print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

    if rank_0():
        save_cfg(cfg, print_cfg=False)

    trainer = F3DTrainer(cfg, device)
    trainer.fit()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
