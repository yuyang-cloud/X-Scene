import logging
import os
import contextlib
from omegaconf import OmegaConf
import itertools
import inspect
import hydra

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers import (
    ModelMixin,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from diffusers.optimization import get_scheduler
from xscene.occ_vae.networks.networks import AutoEncoderGroupSkip
from xscene.runner.occ_validator import OccValidator
from xscene.modules.lora import inject_trainable_lora_extended, save_lora_weight

from ..misc.common import load_module, convert_outputs_to_fp16, move_to
from .base_runner import BaseRunner
from .utils import smart_param_count


class UNetWrapper(ModelMixin):
    """As stated in https://github.com/huggingface/accelerate/issues/668, we
    should not use accumulate provided by accelerator, but create a wrapper to
    two modules.
    """

    def __init__(self, unet, weight_dtype=torch.float32,
                 unet_in_fp16=True) -> None:
        super().__init__()
        self.unet = unet
        self.weight_dtype = weight_dtype
        self.unet_in_fp16 = unet_in_fp16
    
    def prepare(self, cfg, **kwargs):
        self.unet.prepare(cfg, **kwargs)

    def forward(self, noisy_latents, timesteps,
                encoder_hidden_states, encoder_hidden_states_uncond,
                triplane_map_cond, **kwargs):
        kwargs = move_to(
            kwargs, self.weight_dtype, lambda x: x.dtype == torch.float32)


        # Predict the noise residual
        # NOTE: Since we fix most of the model, we cast the model to fp16 and
        # disable autocast to prevent it from falling back to fp32. Please
        # enable autocast on your customized/trainable modules.
        context = contextlib.nullcontext
        context_kwargs = {}
        if self.unet_in_fp16:
            context = torch.cuda.amp.autocast
            context_kwargs = {"enabled": False}
        with context(**context_kwargs):
            model_pred = self.unet(
                noisy_latents,  # b,c,h,w
                timesteps.reshape(-1),  # b
                encoder_hidden_states=encoder_hidden_states.to(dtype=self.weight_dtype) if encoder_hidden_states is not None else None,   # b,len,c 
                encoder_hidden_states_uncond=encoder_hidden_states_uncond.to(dtype=self.weight_dtype) if encoder_hidden_states_uncond is not None else None,  # 1,len,c
                triplane_map_cond=triplane_map_cond.to(dtype=self.weight_dtype) if triplane_map_cond is not None else None,  # b,c,h,w
                bboxes_3d_data=kwargs["bboxes_3d_data"],
            )
        return model_pred


class OccRunner(BaseRunner):
    def __init__(self, cfg, accelerator, train_set, val_set) -> None:
        super().__init__(cfg, accelerator, train_set, val_set)

    def _init_fixed_models(self, cfg):
        # fmt: off
        if self.cfg.model.use_cross_attn_cond:
            self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
            self.tokenizer_t5 = T5Tokenizer.from_pretrained(cfg.model.pretrained_t5_path, model_max_length=512, ignore_mismatched_sizes=True)
            self.text_encoder_t5 = T5EncoderModel.from_pretrained(cfg.model.pretrained_t5_path).requires_grad_(False)

        vae_ckpt_path = cfg.model.vae_ckpt
        if not os.path.isabs(vae_ckpt_path) and not os.path.exists(vae_ckpt_path):
            # If relative path not found, try to resolve it relative to original cwd
            vae_ckpt_path = hydra.utils.to_absolute_path(vae_ckpt_path)
        self.vae = AutoEncoderGroupSkip.from_pretrained(vae_ckpt_path)
        self.noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        # fmt: on

    def _init_trainable_models(self, cfg):
        model_cls = load_module(cfg.model.unet_module)
        unet_param = OmegaConf.to_container(self.cfg.model.unet, resolve=True)
        self.unet = model_cls(**unet_param)

    def _set_model_trainable_state(self, train=True):
        # set trainable status
        if self.cfg.model.use_cross_attn_cond:
            self.text_encoder.requires_grad_(False)
            self.text_encoder_t5.requires_grad_(False)

        self.vae.requires_grad_(False)
        self.unet.train(train)
    
    def _init_validator(self, cfg):
        # validator
        pipe_cls = load_module(cfg.model.pipe_module)
        self.validator = OccValidator(
            self.cfg,
            self.val_dataset,
            # self.train_dataset,
            pipe_cls,
            pipe_param={
                "vae": self.vae,
                "text_encoder": self.text_encoder_t5 if self.cfg.model.use_cross_attn_cond else None,
                "tokenizer": self.tokenizer_t5 if self.cfg.model.use_cross_attn_cond else None,
            }
        )

    def set_optimizer_scheduler(self):
        # optimizer and lr_schedulers
        if self.cfg.runner.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        param_count = smart_param_count(list(self.unet.parameters()))
        logging.info(
            f"[OccRunner] takes {param_count} params from unet to optimizer.")
        params_to_optimize = ([
            {"params": self.unet.parameters()},
        ])
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.cfg.runner.learning_rate,
            betas=(self.cfg.runner.adam_beta1, self.cfg.runner.adam_beta2),
            weight_decay=self.cfg.runner.adam_weight_decay,
            eps=self.cfg.runner.adam_epsilon,
        )

        # lr scheduler
        self._calculate_steps()
        # fmt: off
        self.lr_scheduler = get_scheduler(
            self.cfg.runner.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.runner.lr_warmup_steps * self.cfg.runner.gradient_accumulation_steps,
            num_training_steps=self.cfg.runner.max_train_steps * self.cfg.runner.gradient_accumulation_steps,
            num_cycles=self.cfg.runner.lr_num_cycles,
            power=self.cfg.runner.lr_power,
        )
        # fmt: on

    def prepare_device(self):
        self.unet_wrapper = UNetWrapper(self.unet)
        # accelerator
        ddp_modules = (
            self.unet_wrapper,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        ddp_modules = self.accelerator.prepare(*ddp_modules)
        (
            self.unet_wrapper,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = ddp_modules

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.cfg.model.use_cross_attn_cond:
            self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
            self.text_encoder_t5.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.cfg.runner.unet_in_fp16 and self.weight_dtype == torch.float16:
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        # move optimized params to fp32. TODO: is this necessary?
        if self.cfg.model.use_fp32_for_unet_trainable:
            for name, mod in self.unet.trainable_module.items():
                logging.debug(f"[OccRunner] set {name} to fp32")
                mod.to(dtype=torch.float32)
                mod._original_forward = mod.forward
                # autocast intermediate is necessary since others are fp16
                mod.forward = torch.cuda.amp.autocast(
                    dtype=torch.float16)(mod.forward)
                # we ensure output is always fp16
                mod.forward = convert_outputs_to_fp16(mod.forward)
        else:
            raise TypeError(
                "There is an error/bug in accumulation wrapper, please "
                "make all trainable param in fp32.")
        unet_wrapper = self.accelerator.unwrap_model(self.unet_wrapper)
        unet_wrapper.weight_dtype = self.weight_dtype
        unet_wrapper.unet_in_fp16 = self.cfg.runner.unet_in_fp16

        with torch.no_grad():
            if hasattr(self.unet, "bbox_embedder"):
                self.accelerator.unwrap_model(self.unet).prepare(
                    self.cfg,
                    tokenizer=self.tokenizer,
                    text_encoder=self.text_encoder
                )

        # We need to recalculate our total training steps as the size of the
        # training dataloader may have changed.
        self._calculate_steps()

    def _save_model(self, root=None):
        if root is None:
            root = self.cfg.log_root
        # if self.accelerator.is_main_process:
        unet = self.accelerator.unwrap_model(self.unet)
        unet.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))
        logging.info(f"Save your model to: {root}")

    def _train_one_stop(self, batch):
        self.unet_wrapper.train()
        with self.accelerator.accumulate(self.unet_wrapper):
            triplane_latents = batch["triplane"] # b, c, h, w

            # Sample noise that we'll add to the triplane_latents
            noise = torch.randn_like(triplane_latents)

            bsz = triplane_latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=triplane_latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the triplane_latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self._add_noise(triplane_latents, noise, timesteps)

            # Get the text embedding for conditioning using t5
            if self.cfg.model.use_cross_attn_cond:
                encoder_hidden_states = self.text_encoder_t5(batch["input_ids_t5"])[0].to(dtype=self.weight_dtype)
                encoder_hidden_states_uncond = self.text_encoder_t5(
                    batch["uncond_ids_t5"])[0].to(dtype=self.weight_dtype)
            else:
                encoder_hidden_states = None
                encoder_hidden_states_uncond = None

            # Get the triplane map conditioning
            if self.cfg.model.use_map_cond:
                triplane_map_cond = batch["triplane_map"].to(
                    dtype=self.weight_dtype)
            else:
                triplane_map_cond = None

            model_pred = self.unet_wrapper(
                noisy_latents,  # b,c,h,w
                timesteps.reshape(-1),  # b
                encoder_hidden_states=encoder_hidden_states,   # b,len,c 
                encoder_hidden_states_uncond=encoder_hidden_states_uncond,  # 1,len,c
                triplane_map_cond=triplane_map_cond,  # b,c,h,w
                **batch['kwargs'],
            )

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(
                    triplane_latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction='none')
            loss = loss.mean()

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip = itertools.chain(
                    self.unet_wrapper.parameters(),
                )
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.runner.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(
                set_to_none=self.cfg.runner.set_grads_to_none)

        return loss

    def _validation(self, step):
        unet = self.accelerator.unwrap_model(self.unet)
        occ_logs = self.validator.validate(
            unet, self.accelerator.trackers, step,
            self.weight_dtype, self.accelerator.device)