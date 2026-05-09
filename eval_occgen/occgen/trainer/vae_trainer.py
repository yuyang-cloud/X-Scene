import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

import occgen.utils.constants as C
from occgen.dataset.builder import get_voxel_seq_dataloaders
from occgen.utils.dataset_utils import get_layout, parse_hexplane_path
from occgen.utils.ddp_utils import func_rank_0, rank_0, wandb_log, print_text
from occgen.utils.file_utils import get_latest_ckpt
from occgen.utils.hexplane_utils import compose_featmaps, compose_featmaps_transpose
from occgen.utils.loss_utils import build_losses
from occgen.utils.metrics import Metrics
from occgen.utils.vae_train_utils import get_pred_label, pred_to_voxels

try:
    from occgen.vae.occsora_vae import OccSoraVAE
    from occgen.vae.vae import DynamicAE
except ImportError:
    OccSoraVAE = None
    DynamicAE = None

warnings.filterwarnings('ignore')


class VAETrainer:
    def __init__(self, cfg, device):
        # model independent cfg
        self.cfg = cfg
        self.device = device
        self.amp = cfg.trainer.amp
        self.num_classes = cfg.dataset.num_classes
        self.num_frames = cfg.dataset.t_length
        self.log_frame_miou = cfg.model.log_frame_miou
        self.log_class_miou = cfg.model.log_class_miou
        self.log_train_miou = cfg.model.log_train_miou
        self.iou_class_names = list(cfg.dataset.label_to_names.values())
        self.log_freq = cfg.trainer.log_interval
        self.num_epochs = cfg.trainer.num_epochs
        self.decoder_type = cfg.model.get('decoder_type', '')

        # model dependent cfg
        self.occsora = cfg.get('occsora', False)
        self.interpolate_t_skip = cfg.model.get('interpolate_t_skip', 0)

        # setup dataloaders
        self.train_loader, self.valid_loader = get_voxel_seq_dataloaders(cfg, shuffle=cfg.dataset.shuffle)

        # load model
        model_class = OccSoraVAE if self.occsora else DynamicAE
        if model_class is None:
            raise ImportError("The sequence VAE model is not included in this package.")
        self.model = model_class(cfg)

        # setup ddp
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.device])

        print_text(f'VAE Parameters: {sum(p.numel() for p in self.model.parameters()):,}')

        # setup loss
        self.loss_history = dict()
        self.loss_names = list(cfg.model.loss_weights.keys())
        self.loss_weights = dict(**cfg.model.loss_weights)
        self.loss_functions = build_losses(cfg.model, cfg=cfg, device=self.device)

        # setup optimizer
        self.optimizer, self.scheduler = self.setup_optimizer()
        if self.amp:
            self.scaler = GradScaler()

        # setup metrics
        self.metrics = self.setup_metrics()

        # counters
        self.global_step = 0
        self.current_epoch = 0
        self.best_miou = 0.0

        # resume
        self.load_checkpoint(self.get_resume_ckpt())

    def setup_metrics(self):
        metric_names = ['valid']
        if self.log_train_miou:
            metric_names += ['train']
        if self.log_frame_miou:
            metric_names += [f'valid_{t}' for t in range(self.num_frames)]
        if self.log_frame_miou and self.log_train_miou:
            metric_names += [f'train_{t}' for t in range(self.num_frames)]
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

        if self.cfg.model.scheduler_type == 'multisteplr':
            sch = MultiStepLR(opt, self.cfg.model.lr_scheduler_steps, self.cfg.model.lr_scheduler_decay)
        elif self.cfg.model.scheduler_type == 'cosinelrscheduler':
            sch = CosineLRScheduler(
                opt, t_initial=len(self.train_loader) * self.num_epochs, lr_min=1e-6,
                warmup_t=self.cfg.model.warmup_iters, warmup_lr_init=1e-6, t_in_epochs=False
            )
        elif self.cfg.model.scheduler_type == 'cosineannealinglr':
            sch = CosineAnnealingLR(
                opt, T_max=self.num_epochs // self.cfg.model.t_max_ratio,
                eta_min=self.cfg.model.eta_min
            )
        else:
            sch = None

        return opt, sch

    def get_resume_ckpt(self):
        ckpt_path = None
        if self.cfg.trainer.resume_ckpt is not None:
            ckpt_path = self.cfg.trainer.resume_ckpt
        elif self.cfg.trainer.auto_resume:
            ckpt_path = get_latest_ckpt(C.CKPT_PATH, self.cfg.name)
        return ckpt_path

    @func_rank_0
    def save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
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
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_miou = checkpoint['best_miou']
        self.global_step = checkpoint['global_step']

        if rank_0():
            print(f'Loaded checkpoint {checkpoint_path}')

    def add_to_metrics(self, pred, gt, invalid, name='train'):
        metrics = self.metrics[name]
        pred = get_pred_label(pred)
        masks = invalid != 1
        eval_output = pred[masks]
        eval_label = gt[masks]
        return metrics.add_batch(eval_output.to(torch.int32), eval_label.to(torch.int32))

    def log_metrics_and_reset(self, name='train'):
        metrics = self.metrics[name]
        iou, miou_all, miou_sem, _ = metrics.get_metrics_dist()
        metrics.reset()

        wandb_log({f'{name}/mIoU_sem': miou_sem}, step=self.global_step)

        if self.log_class_miou:
            for class_idx, class_name in enumerate(self.iou_class_names):
                wandb_log({f'{name}/{class_idx + 1}_{class_name}': iou[class_idx]}, step=self.global_step)

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
            loss_params = None,
            if loss_name == 'ce':
                if len(pred.shape) > 3:
                    loss_params = pred.permute(0, 5, 1, 2, 3, 4), gt
                else:
                    loss_params = pred.reshape(-1, self.num_classes), gt.reshape(-1, )
            elif loss_name == 'lovasz':
                loss_params = pred.reshape(-1, self.num_classes), gt.reshape(-1, )
            elif loss_name == 'kl':
                loss_params = model_out['mus'], model_out['logvars']
            elif loss_name == 'embed':
                loss_params = model_out['embed_loss'],
            losses[loss_name] = loss_func(*loss_params)
        return losses

    def step(self, vox, *args, **kwargs):
        vox = vox[:, ::self.interpolate_t_skip + 1]

        # forward pass
        model_out = self.model(vox, *args, **kwargs)

        # loss
        pred = model_out['pred']
        if self.decoder_type == 'query':
            gt = kwargs['query_gt']
        else:
            gt = vox
        losses = self.compute_losses(pred, gt, model_out)
        losses['loss'] = sum([self.loss_weights[loss_name] * losses[loss_name] for loss_name in losses.keys()])

        return {
            'losses': losses,
            'out': pred,
        }

    def forward_step(self, batch, name='train'):
        vox, invalid = batch['voxel'], batch['invalid']
        vox, invalid = vox.to(self.device), invalid.to(self.device)
        vox = vox[..., ::4, ::4, ::4]
        invalid = invalid[..., ::4, ::4, ::4]

        if self.decoder_type == 'query':
            query, query_gt = batch['queries'], batch['query_gts']
            query, query_gt = query.to(self.device), query_gt.to(self.device)
            kwargs = {
                'query': query,
                'query_gt': query_gt
            }
        else:
            kwargs = dict()

        # forward and loss
        if self.amp:
            with autocast():
                step_out = self.step(vox, **kwargs)
        else:
            step_out = self.step(vox, **kwargs)

        # add to epoch level history
        losses = step_out['losses']
        for loss_name in losses.keys():
            self.loss_history[name][loss_name] = self.loss_history[name][loss_name] + (losses[loss_name])

        if name == 'train' and self.global_step % self.log_freq == 0:
            wandb_log({'step/loss': losses['loss']}, step=self.global_step)
            wandb_log({'step/cuda': torch.cuda.memory_reserved() / 2 ** 30}, step=self.global_step)

        # eval
        if self.log_train_miou or name == 'valid':
            pred = step_out['out']

            if self.decoder_type == 'query':
                pred = pred_to_voxels(pred, batch['queries_int'], self.cfg.dataset.grid_size, self.num_frames)

            _, miou_all, miou_sem, _ = self.add_to_metrics(pred, vox, invalid, name=name)
            if name == 'train' and self.global_step % self.log_freq == 0:
                wandb_log({'step/mIoU_sem': miou_sem}, step=self.global_step)

            if self.log_frame_miou:
                for t in range(self.num_frames):
                    frame_pred = pred[:, t]
                    frame_vox = vox[:, t].unsqueeze(1)
                    frame_invalid = invalid[:, t].unsqueeze(1)
                    frame_name = f'{name}_{t}'
                    self.add_to_metrics(frame_pred, frame_vox, frame_invalid, frame_name)

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
        if self.log_train_miou or name == 'valid':
            self.log_metrics_and_reset(name)
            if self.log_frame_miou:
                for t in range(self.num_frames):
                    self.log_metrics_and_reset(f'{name}_{t}')

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
            # forward
            step_out = self.forward_step(batch, name='train')
            loss = step_out['losses']['loss']

            # backward
            self.backward_step(loss)

            if self.cfg.model.scheduler_type == 'cosinelrscheduler':
                self.scheduler.step_update(self.global_step)

            lr = self.optimizer.param_groups[0]['lr']
            if self.global_step % self.log_freq == 0:
                wandb_log({'lr': lr}, step=self.global_step)

            self.global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        if self.cfg.model.scheduler_type in ['multisteplr', 'cosineannealinglr']:
            self.scheduler.step()

        epoch_end_time = time.time()
        wandb_log({'epoch_duration': epoch_end_time - epoch_start_time}, step=self.global_step)
        wandb_log({'train_steps_per_sec': (epoch_end_time - epoch_start_time) / len(self.train_loader)}, step=self.global_step)
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
            val_miou = self.metrics['valid'].get_metrics()[2]
            if self.best_miou <= val_miou:
                self.save_checkpoint(
                    Path(C.CKPT_PATH) / str(self.cfg.name) /
                    C.CKPT_FILENAME_RULE.format(self.current_epoch, val_miou)
                )
            self.best_miou = max(val_miou, self.best_miou)
            self.train_val_epoch_end(name='valid')
        else:
            iou, _, miou, _ = self.metrics['valid'].get_metrics_dist()
            if rank_0():
                print(f'Validation mIoU: {miou}')
                for class_idx, class_name in enumerate(self.iou_class_names):
                    print(f'Class {class_idx + 1} {class_name} IoU: {iou[class_idx]}')

    def fit(self):
        # initialize loggers
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

    # inference
    def predict(self, save_latent, save_voxel, save_layout, new_rollout):
        self.model.eval()
        with torch.no_grad():
            # self.predict_epoch(self.train_loader, 'train', save_latent, save_voxel, save_layout, new_rollout)
            self.predict_epoch(self.valid_loader, 'valid', save_latent, save_voxel, save_layout, new_rollout)

    def predict_epoch(self, loader, name, save_latent, save_voxel, save_layout, new_rollout):
        assert loader.batch_size == 1, 'inference batch size must be 1'
        pbar = tqdm(loader, desc=f"{name}", disable=not rank_0(), leave=False)
        for i, batch in enumerate(pbar):

            # batch['voxel'][:, :-1] = batch['voxel'][:, -1]
            # batch['invalid'][:, :-1] = batch['invalid'][:, -1]

            pred, vox, latent, path = self.predict_batch(batch)
            if save_latent:
                latent_folder = Path(C.HEXPLANE_PATH) / self.cfg.dataset.dataset / self.cfg.name
                latent_folder.mkdir(parents=True, exist_ok=True)  # hexplane/carlasc/0000_exp
                self.save_batch_latent(latent, path, latent_folder, new_rollout)
            if save_voxel:
                voxel_folder = (Path(C.RECONSTRUCT_PATH) / C.VOXEL_PATH / self.cfg.name /
                                f'{name}-{i + dist.get_rank() * len(loader)}')
                voxel_folder.mkdir(parents=True, exist_ok=True)
                self.save_batch_voxel(pred, vox, voxel_folder)
            if save_layout is not None:
                layout_folder = Path(C.LAYOUT_PATH) / self.cfg.dataset.dataset / str(self.num_frames)
                layout_folder.mkdir(parents=True, exist_ok=True)
                self.save_batch_layout(vox, path, layout_folder, save_layout)

    def predict_batch(self, batch):
        paths = batch['paths']
        vox, invalid = batch['voxel'], batch['invalid']
        vox, invalid = vox.to(self.device), invalid.to(self.device)

        if self.decoder_type == 'query':
            query = batch['query']
            query = query.to(self.device)
            kwargs = {
                'query': query,
            }
        else:
            kwargs = dict()

        skipped_vox = vox[:, ::self.interpolate_t_skip + 1]

        if self.amp:
            with autocast():
                batch_out = self.model(skipped_vox, **kwargs)
        else:
            batch_out = self.model(skipped_vox, **kwargs)

        return batch_out['pred'], vox, batch_out['latent'], paths[0][0]

    def save_batch_latent(self, latent, path, folder, new_rollout):
        if self.occsora:
            feat_maps = latent[0].cpu().numpy()
        else:
            feat_xy, feat_xz, feat_yz, feat_tx, feat_ty, feat_tz = latent
            txyz = feat_tx.shape[2:] + feat_yz.shape[2:]
            compose = compose_featmaps_transpose if new_rollout else compose_featmaps
            feat_maps = compose(*latent, txyz).cpu().numpy()
        save_path = folder / parse_hexplane_path(self.cfg.dataset, path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, feat_maps)

    def save_batch_voxel(self, pred, vox, folder):
        pred = torch.softmax(pred.to(torch.float32), dim=-1)
        pred = pred.argmax(dim=-1)
        pred = pred.cpu().numpy().astype(np.uint8)
        vox = vox.cpu().numpy().astype(np.uint8)
        for t in range(vox.shape[1]):
            pred[:, t].tofile(folder / f'{t}_pred.label')
            vox[:, t].tofile(folder / f'{t}_orig.label')

    def save_batch_layout(self, vox, path, layout_folder, down_size):
        layout = get_layout(vox.squeeze(0), down_size).cpu().numpy().astype(np.uint8)
        save_path = layout_folder / parse_hexplane_path(self.cfg.dataset, path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, layout)
