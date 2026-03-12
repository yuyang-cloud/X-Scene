from torch.utils.tensorboard import SummaryWriter
from xscene.occ_vae.dataset.nuscenes_occ_dataset import NuScenesDatasetOcc
from xscene.occ_vae.networks.networks import AutoEncoderGroupSkip
from xscene.occ_vae.loss.lovasz import lovasz_softmax
from xscene.occ_vae.utils.ssc_metrics import SSCMetrics
from xscene.occ_vae.utils.utils import point2voxel_class
from xscene.occ_vae.utils.dist_utils import reduce_tensor
from tools.vis_occ import visualize_occ
import shutil
import math
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np

import torch.distributed as dist
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler


def dataset_builder(args):
    print("build dataset")
    if args.dataset == 'Occ3D-nuScenes':
        # range=[-40m,-40m,-1m,40m,40m,5.4m] voxel_size=[0.4m,0.4m,0.4m] volume_size=[200,200,16]
        args.num_class = 18
        args.grid_size = [200, 200, 16]
        args.pc_range = [-40, -40, -1, 40, 40, 5.4]
        dataset = NuScenesDatasetOcc(args, 'train')
        val_dataset = NuScenesDatasetOcc(args, 'val')
        class_names = [
            'others',
            'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
            'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
            'terrain', 'manmade', 'vegetation'
        ]
    elif args.dataset == 'nuScenes-Occupancy':
        args.num_class = 18
        args.grid_size = [512, 512, 40]
        args.pc_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
        dataset = NuScenesDatasetOcc(args, 'train')
        val_dataset = NuScenesDatasetOcc(args, 'val')
        class_names = [
            'noise',
            'barrier', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'pedestrian',
            'trafficcone', 'trailer', 'truck', 'driveable_surface', 'other', 'sidewalk',
            'terrain', 'manmade', 'vegetation'
        ]

    return dataset, val_dataset, args.num_class, class_names

class Trainer:
    def __init__(self, args, rank, distributed):
        self.args = args
        self.rank = rank
        self.distributed = distributed
        self.writer = SummaryWriter(os.path.join(args.save_path, 'tb')) if rank==0 else None
        self.total_epoch = args.total_epoch
        self.cur_epoch = 0
        self.global_step = 0
        self.best_iou = 0
        self.best_miou = 0
        self.best_iou_trainer_ckpt_pth = ""
        self.best_iou_model_ckpt_pth = ""
        self.best_miou_trainer_ckpt_pth = ""
        self.best_miou_model_ckpt_pth = ""

        # dataset
        self.train_dataset, self.val_dataset, self.num_class, class_names = dataset_builder(args)
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False, drop_last=False)
        else:
            self.train_sampler, self.val_sampler = None, None
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=args.bs, sampler=self.train_sampler, num_workers=12, pin_memory=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, sampler=self.val_sampler, num_workers=8, pin_memory=True)
        self.iou_class_names = class_names

        # model
        if args.resume:
            raw_model = AutoEncoderGroupSkip.from_pretrained(args.resume + '_model')
        else:
            raw_model = AutoEncoderGroupSkip(
                num_class = args.num_class,
                geo_feat_channels = args.geo_feat_channels,
                feat_channel_up = args.feat_channel_up,
                mlp_hidden_channels = args.mlp_hidden_channels,
                mlp_hidden_layers = args.mlp_hidden_layers,
                padding_mode = args.padding_mode,
                z_down = args.z_down,
                xy_down = args.xy_down,
                xy_down_times = args.xy_down_times,
                use_vae = args.use_vae,
                voxel_fea = args.voxel_fea,
                triplane = args.triplane,
                use_deform_attn = args.use_deform_attn,
                pos = args.pos,
                dataset = args.dataset,
            )
        if self.distributed:
            raw_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                raw_model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=False
            )
            self.raw_model = self.model.module
        else:
            self.model = raw_model.cuda()
            self.raw_model = self.model

        #  optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        if args.steplr_scheduler:
            self.scheduler = MultiStepLRScheduler(self.optimizer, args.lr_scheduler_steps, args.lr_scheduler_decay)
        else:
            self.scheduler = CosineLRScheduler(self.optimizer,
                                               t_initial=len(self.train_dataloader) * args.total_epoch,
                                               lr_min=1e-6,
                                               warmup_t=500,
                                               warmup_lr_init=1e-6,
                                               t_in_epochs=False)
        self.grad_scaler = GradScaler()
        
        if args.resume:
            # load trainer
            trainer_ckpt_path = args.resume + '_trainer.pt'
            checkpoint = torch.load(trainer_ckpt_path)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.cur_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step'] 
            if 'best_val_iou' in checkpoint:
                self.best_iou = checkpoint['best_iou']
            if 'best_val_miou' in checkpoint:
                self.best_miou = checkpoint['best_miou']

        # loss functions
        self.loss_fns = {}
        self.loss_fns['ce'] = torch.nn.CrossEntropyLoss(weight=self.train_dataset.weights, ignore_index=255)
        self.loss_fns['lovasz'] = None
        if self.raw_model.use_vae:
            self.loss_fns['kl'] = None

    def train(self):
        while self.cur_epoch < self.total_epoch:

            if self.rank == 0:
                print(f'Training Epoch [{self.cur_epoch}]...')
            self._train_model()
            
            if self.cur_epoch % self.args.eval_epoch == 0 and self.rank == 0:
                print(f'Evaluation Epoch [{self.cur_epoch}]...')
            self._eval_and_save_model()

            self.scheduler.step(self.cur_epoch)
            if self.rank == 0:
                self.writer.add_scalar('lr_epochwise', self.optimizer.param_groups[0]['lr'], global_step=self.cur_epoch)
            self.cur_epoch += 1

    def compute_lovasz_loss(self, output_voxels, target_voxels):
        lovasz_loss = lovasz_softmax(output_voxels, target_voxels, ignore=255)
        return lovasz_loss

    def _loss(self, vox, query, label, losses, coord):
        empty_label = 0.
        # forward pass
        if self.raw_model.use_vae:
            preds, means, logvars = self.model(vox, query)
        else:
            preds = self.model(vox, query) # [bs, N, 20]

        # ce loss
        losses['ce'] = self.loss_fns['ce'](preds.view(-1, self.num_class), label.view(-1,))
        losses['loss'] = losses['ce']

        # KL loss
        if self.raw_model.use_vae:
            kl_losses = []
            for mean, logvar in zip(means, logvars):
                kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                kl_losses.append(kl)
            kl_loss = sum(kl_losses)
            kl_weight = self.get_kl_weight_sigmoid(self.global_step)
            losses['kl'] = kl_weight * kl_loss
            losses['loss'] += losses['kl']
        
        # lovasz loss
        pred_output = torch.full((preds.shape[0], vox.shape[1], vox.shape[2], vox.shape[3], self.num_class), fill_value=empty_label, device=preds.device, dtype=preds.dtype)
        gt_output = torch.full((preds.shape[0], vox.shape[1], vox.shape[2], vox.shape[3]), fill_value=empty_label, device=preds.device)
        for i in range(preds.shape[0]):
            pred_output[i, coord[i, :, 0], coord[i, :, 1], coord[i, :, 2], :] = preds[i]    # B,H,W,D,cls
            gt_output[i, coord[i, :, 0], coord[i, :, 1], coord[i, :, 2]] = label[i].float()         # B,H,W,D
        pred_output = pred_output.permute(0,4,1,2,3)    # B,cls,H,W,D

        if self.args.dataset == 'nuScenes-Occupancy':
            pred_output, gt_output, _ = self.downsample_voxels(pred_output, gt_output)

        pred_output = torch.nn.functional.softmax(pred_output, dim=1)
        losses['lovasz'] = self.compute_lovasz_loss(pred_output, gt_output)
        losses['loss'] += losses['lovasz']

        adaptive_weight = None
        return losses, preds, adaptive_weight
    
    def get_kl_weight_sigmoid(self, step, max_weight=0.01, center_step=10000, steepness=5e-4):
        """
        Sigmoid annealing schedule for KL weight.
        
        Args:
            step (int): Current global training step.
            max_weight (float): Maximum KL weight (e.g., 0.01).
            center_step (int): The step where the sigmoid crosses 0.5 * max_weight.
            steepness (float): Controls how sharp the transition is.
        
        Returns:
            float: KL weight at current step.
        """
        return float(max_weight / (1 + np.exp(-steepness * (step - center_step))))
    
    def downsample_voxels(self, output_voxels, target_voxels):
        """
            output_voxels = inter_num, bs, c, h,w,d
            target_voxels =            bs, H,W,D
        """
        B, C, pH, pW, pD = output_voxels.shape
        tB, tH, tW, tD = target_voxels.shape

        H, W, D = 256, 256, 20
        # output_voxel align to H,W,D
        if pH != H:
            output_voxels = F.interpolate(output_voxels, size=(H, W, D), mode='trilinear', align_corners=False)
        
        # target_voxel align to H,W,D
        ratio = tH // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_idx = 0
            empty_mask = target_voxels.sum(-1) == empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()
            invalid = target_voxels == 255
        return output_voxels, target_voxels, invalid

    def _train_model(self):
        self.model.train()
        os.environ['eval'] = 'false'

        total_losses = {loss_name: 0. for loss_name in self.loss_fns.keys()}
        total_losses['loss'] = 0.
        evaluator = SSCMetrics(self.num_class, []) if self.args.eval_during_training else None
        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.cur_epoch)
        if self.rank == 0:
            dataloader_tqdm = tqdm(self.train_dataloader)
        else:
            dataloader_tqdm = self.train_dataloader

        for vox, query, label, coord, path, invalid in dataloader_tqdm:
            vox = vox.type(torch.LongTensor).cuda()         # [256,256,3] vox_label
            query = query.type(torch.FloatTensor).cuda()    # N,3 [-1,1] query_coord
            label = label.type(torch.LongTensor).cuda()     # N,1       query_label
            coord = coord.type(torch.LongTensor).cuda()     # N,3       query_voxcoord
            invalid = invalid.type(torch.LongTensor).cuda() # [256,256,3]
            b_size = vox.size(0)

            # forward
            losses = {}
            with autocast():
                losses, model_output, adaptive_weight = self._loss(vox, query, label, losses, coord)

            # optimize
            self.optimizer.zero_grad()
            self.grad_scaler.scale(losses['loss']).backward()
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # gradient clipping
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # learning rate scheduling
            self.scheduler.step_update(self.global_step)

            if self.args.eval_during_training:
                # point2voxel
                output = point2voxel_class(self.args, model_output, coord)   # B,H,W,D,cls
                output = output.permute(0,4,1,2,3)  # B,cls,H,W,D

                # downsample
                if self.args.dataset == 'nuScenes-Occupancy':
                    output, vox, invalid = self.downsample_voxels(output.detach(), vox.detach())

                # eval and log each iteration
                output = get_pred_mask(output)
                masks = evaluator.get_eval_mask(vox, invalid)
                eval_output = output[masks]
                eval_label = vox[masks]
                this_iou, this_miou = evaluator.addBatch(eval_output.cpu().numpy().astype(int), eval_label.cpu().numpy().astype(int))
                this_iou = torch.tensor(this_iou).to(eval_output)
                this_miou = torch.tensor(this_miou).to(eval_output)

            # reduce ranks and log
            torch.cuda.synchronize()
            if self.global_step % self.args.display_period == 0:

                if self.distributed:
                    for loss_name in losses.keys():
                        losses[loss_name] = reduce_tensor(losses[loss_name], self.args.world_size)
                    if self.args.eval_during_training:
                        this_iou = reduce_tensor(this_iou, self.args.world_size)
                        this_miou = reduce_tensor(this_miou, self.args.world_size)
                
                if self.rank == 0:
                    # on display
                    if self.args.eval_during_training:
                        dataloader_tqdm.set_postfix({"loss": losses['loss'].detach().item(), "iou": this_iou.detach().item(), "miou": this_miou.detach().item()})
                    else:
                        dataloader_tqdm.set_postfix({"loss": losses['loss'].detach().item()})

                    # on tensorboard
                    self.writer.add_scalar('Grad_Norm', grad_norm, global_step=self.global_step)
                    for loss_name in losses.keys():
                        self.writer.add_scalar(f'Train_Loss_stepwise/loss_{loss_name}', losses[loss_name], self.global_step)
                    if self.args.eval_during_training:
                        self.writer.add_scalar('Train_Performance_stepwise/IoU', this_iou, global_step=self.global_step)
                        self.writer.add_scalar('Train_Performance_stepwise/mIoU', this_miou, global_step=self.global_step)
          
            # loss accumulation for logging
            for loss_name in losses.keys():
                total_losses[loss_name] += (losses[loss_name] * b_size)

            self.global_step += 1

        # eval for 1 epoch
        if self.args.eval_during_training:
            _, class_jaccard = evaluator.getIoU()
            m_jaccard = class_jaccard[1:].mean()
            miou = m_jaccard * 100
            conf = evaluator.get_confusion()
            iou = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0] + 1e-8)
            class_jaccard = torch.tensor(class_jaccard).to(eval_output)
            iou = torch.tensor(iou).to(eval_output)
            miou = torch.tensor(miou).to(eval_output)
            evaluator.reset()

        # reduce ranks
        torch.cuda.synchronize()
        if self.distributed:
            for loss_name in total_losses.keys():
                total_losses[loss_name] = reduce_tensor(total_losses[loss_name], self.args.world_size)
            if self.args.eval_during_training:
                class_jaccard = reduce_tensor(class_jaccard, self.args.world_size)
                iou = reduce_tensor(iou, self.args.world_size)
                miou = reduce_tensor(miou, self.args.world_size)

        # log for 1 epoch
        if self.rank == 0:
            for loss_name in losses.keys():
                self.writer.add_scalar(f'Train_Loss_epochwise/loss_{loss_name}', total_losses[loss_name] / len(self.train_dataset), global_step=self.cur_epoch)
            if self.args.eval_during_training:
                self.writer.add_scalar('Train_Performance_epochwise/IoU', iou, global_step=self.cur_epoch)
                self.writer.add_scalar('Train_Performance_epochwise/mIoU', miou, global_step=self.cur_epoch)
                for class_idx, class_name in enumerate(self.iou_class_names):
                    self.writer.add_scalar(f'Train_ClassPerformance_epochwise/class{class_idx + 1}_IoU_{class_name}', class_jaccard[class_idx + 1], global_step=self.cur_epoch)
                print(f"Train_Performance Epoch [{self.cur_epoch}] \t IOU: \t {iou:01f} \t mIoU: \t {miou:01f}")


    @torch.no_grad()
    def _eval_and_save_model(self):
        self.model.eval()
        os.environ['eval'] = 'true'

        total_losses = {loss_name: 0. for loss_name in self.loss_fns.keys()}
        total_losses['loss'] = 0.
        evaluator = SSCMetrics(self.num_class, [])
        if self.rank == 0:
            dataloader_tqdm = tqdm(self.val_dataloader)
        else:
            dataloader_tqdm = self.val_dataloader

        with torch.no_grad():
            for sample_idx, (vox, query, label, coord, path, invalid) in enumerate(dataloader_tqdm):
                vox = vox.type(torch.LongTensor).cuda()
                query = query.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                coord = coord.type(torch.LongTensor).cuda()
                invalid = invalid.type(torch.LongTensor).cuda()
                b_size = vox.size(0)  # TODO: check correctness

                losses = {}
                losses, model_output, adaptive_weight = self._loss(vox, query, label, losses, coord)
                
                # point2voxel
                output = point2voxel_class(self.args, model_output, coord)   # B,H,W,D,cls
                output = output.permute(0,4,1,2,3)  # B,cls,H,W,D

                # downsample
                if self.args.dataset == 'nuScenes-Occupancy':
                    output, vox, invalid = self.downsample_voxels(output.detach(), vox.detach())

                # eval and log each iteration
                output = get_pred_mask(output)
                masks = evaluator.get_eval_mask(vox, invalid)

                eval_output = output[masks]
                eval_label = vox[masks]
                this_iou, this_miou = evaluator.addBatch(eval_output.cpu().numpy().astype(int), eval_label.cpu().numpy().astype(int))
                this_iou = torch.tensor(this_iou).to(eval_output)
                this_miou = torch.tensor(this_miou).to(eval_output)

                # reduce ranks
                torch.cuda.synchronize()
                if self.distributed:
                    for loss_name in losses.keys():
                        losses[loss_name] = reduce_tensor(losses[loss_name], self.args.world_size)
                    this_iou = reduce_tensor(this_iou, self.args.world_size)
                    this_miou = reduce_tensor(this_miou, self.args.world_size)
                
                # on display
                if self.rank == 0:
                    dataloader_tqdm.set_postfix({"loss": losses['loss'].detach().item(), "iou": this_iou.detach().item(), "miou": this_miou.detach().item()})

                for loss_name in losses.keys():
                    total_losses[loss_name] += (losses[loss_name] * b_size)

                # idx = path[0].split('/')[-1].split('.')[0]
                # folder = path[0].split('/')[-3]
                # save_remap_lut(self.args, output, folder, idx, self.train_dataset.learning_map_inv, True)

        # eval for all validation samples
        _, class_jaccard = evaluator.getIoU()
        m_jaccard = class_jaccard[1:].mean()
        miou = m_jaccard * 100
        conf = evaluator.get_confusion()
        iou = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0] + 1e-8)
        class_jaccard = torch.tensor(class_jaccard).to(eval_output)
        iou = torch.tensor(iou).to(eval_output)
        miou = torch.tensor(miou).to(eval_output)
        evaluator.reset()

        # reduce ranks
        torch.cuda.synchronize()
        if self.distributed:
            for loss_name in total_losses.keys():
                total_losses[loss_name] = reduce_tensor(total_losses[loss_name], self.args.world_size)
            class_jaccard = reduce_tensor(class_jaccard, self.args.world_size)
            iou = reduce_tensor(iou, self.args.world_size)
            miou = reduce_tensor(miou, self.args.world_size)

        # log and save ckpt
        if self.rank == 0:
            self.writer.add_scalar('Val_Performance_epochwise/IoU', iou, global_step=self.cur_epoch)
            self.writer.add_scalar('Val_Performance_epochwise/mIoU', miou, global_step=self.cur_epoch)
            for class_idx, class_name in enumerate(self.iou_class_names):
                self.writer.add_scalar(f'Val_ClassPerformance_epochwise/class{class_idx + 1}_IoU_{class_name}', class_jaccard[class_idx + 1], global_step=self.cur_epoch)
            for loss_name in losses.keys():
                self.writer.add_scalar(f'Val_Loss_epochwise/loss_{loss_name}', total_losses[loss_name] / len(self.val_dataset), global_step=self.cur_epoch)
            print(f"Val_Performance Epoch [{self.cur_epoch}] \t IOU: \t {iou:01f} \t mIoU: \t {miou:01f} \n")

            if self.best_iou < iou:
                if os.path.exists(self.best_iou_trainer_ckpt_pth):  # delet old ckpt
                    os.remove(self.best_iou_trainer_ckpt_pth)
                if os.path.exists(self.best_iou_model_ckpt_pth):
                    safe_remove(self.best_iou_model_ckpt_pth)
                self.best_iou = iou
                # save trainer
                self.best_iou_trainer_ckpt_pth = self.args.save_path + "/" + str(self.cur_epoch) + "_iou=" + str(f"{iou:.3f}") + '_trainer.pt'
                checkpoint = {'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(), 'epoch': self.cur_epoch, 'global_step': self.global_step, 'best_iout': self.best_iou, 'best_miou': self.best_miou}
                # torch.save(checkpoint, self.best_iou_trainer_ckpt_pth)
                # save model
                self.best_iou_model_ckpt_pth = self.args.save_path + "/" + str(self.cur_epoch) + "_iou=" + str(f"{iou:.3f}") + '_model'
                # self.raw_model.save_pretrained(self.best_iou_model_ckpt_pth)
            if self.best_miou < miou:
                if os.path.exists(self.best_miou_trainer_ckpt_pth):
                    os.remove(self.best_miou_trainer_ckpt_pth)
                if os.path.exists(self.best_miou_model_ckpt_pth):
                    safe_remove(self.best_miou_model_ckpt_pth)
                self.best_miou = miou
                # save trianer
                self.best_miou_trainer_ckpt_pth = self.args.save_path + "/" + str(self.cur_epoch) + "_miou=" + str(f"{miou:.3f}") + '_trainer.pt'
                checkpoint = {'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(), 'epoch': self.cur_epoch, 'global_step': self.global_step, 'best_iout': self.best_iou, 'best_miou': self.best_miou}
                torch.save(checkpoint, self.best_miou_trainer_ckpt_pth)
                # save model
                self.best_miou_model_ckpt_pth = self.args.save_path + "/" + str(self.cur_epoch) + "_miou=" + str(f"{miou:.3f}") + '_model'
                self.raw_model.save_pretrained(self.best_miou_model_ckpt_pth)
                # self.raw_model._config.save_pretrained(self.best_miou_model_ckpt_pth)

def get_pred_mask(model_output, separate_decoder=False):
    preds = model_output
    pred_prob = torch.softmax(preds, dim=1)
    pred_mask = pred_prob.argmax(dim=1).float()
    return pred_mask

def safe_remove(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        print(f"Path {path} does not exist.")