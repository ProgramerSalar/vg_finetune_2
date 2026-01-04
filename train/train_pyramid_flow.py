import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import datetime
import numpy as np
import time
import torch
import logging
import json
import math
import random
import diffusers
import transformers
from pathlib import Path
from packaging import version
from copy import deepcopy




from dataset.dataset import LengthGroupedVideoTextDataset
from dataset.dataloader import create_length_grouped_video_text_dataloader

from pipeline import PyramidDiTForVideoGeneration
from flux.flux import FluxSingleTransformerBlock, FluxTransformerBlock
from args import get_args



from trainer_misc import (
    create_optimizer,
    train_one_epoch_with_fsdp,
    constant_scheduler,
    cosine_scheduler,
)



from collections import OrderedDict

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, 
    FullStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
    CPUOffload,
    StateDictType,
)

from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.t5.modeling_t5 import T5Block

import accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate import FullyShardedDataParallelPlugin
from diffusers.utils import is_wandb_available
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from diffusers.optimization import get_scheduler

logger = get_logger(__name__)



def build_model_runner(args):

  model_dtype = args.model_dtype
  model_path = args.model_path
  model_name = args.model_name 
  model_varient = args.model_variant

  print(f"Load the {model_name} model checkpoint from path: {model_path}, using dtype {model_dtype}")
  sample_ratios=[1, 2, 1]
  assert args.batch_size % int(sum(sample_ratios)) == 0, "The batch_size should be divisible by sum(sample_ratios)"
  



  runner = PyramidDiTForVideoGeneration(
    model_path = model_path,
    model_dtype=model_dtype,
    model_name = model_name, 
    use_gradient_checkpointing=args.gradient_checkpointing,
    gradient_checkpointing_ratio=args.gradient_checkpointing_ratio,
    return_log=True,
    model_varient=model_varient,
    timestep_shift=args.schedule_shift,
    stages=[1, 2, 4],
    stage_range=[0, 1/3, 2/3, 1],
    sample_ratios=sample_ratios,
    use_mixed_training=True,
    use_flash_attn=args.use_flash_attn,
    load_text_encoder=args.load_text_encoder,
    load_vae=args.load_vae,
    max_temporal_length=args.max_frames,
    frame_per_unit=args.frame_per_unit,
    use_temporal_causal=args.use_temporal_causal,
    corrupt_ratio=args.corrupt_ratio,
    interp_condition_pos=args.interp_condition_pos,
    video_sync_group=args.video_sync_group

  )

  return runner

def auto_resume(args, accelerator):
  if len(args.resume) > 0:
      path = args.resume
  else:
      # Get the most recent checkpoint
      dirs = os.listdir(args.output_dir)
      dirs = [d for d in dirs if d.startswith("checkpoint")]
      dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
      path = dirs[-1] if len(dirs) > 0 else None

  if path is None:
      accelerator.print(
          f"Checkpoint does not exist. Starting a new training run."
      )
      initial_global_step = 0
  else:
      accelerator.print(f"Resuming from checkpoint {path}")
      accelerator.load_state(os.path.join(args.output_dir, path))
      global_step = int(path.split("-")[1])
      initial_global_step = global_step
  
  return initial_global_step


def main(args):

  logging_dir = Path(args.output_dir, args.logging_dir)
  
  accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir,
                                                    logging_dir=logging_dir)
  
  accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.model_dtype,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        
    )

  # REMOVED: Sequence Parallel logic (Not needed for Single GPU)

  if args.report_to == "wandb":
      if not is_wandb_available():
          raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
      import wandb

  # Keep logging setup (It's useful for debugging)
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
  )
  logger.info(accelerator.state, main_process_only=False)

  # Keep verbosity settings (Cleaner logs)
  if accelerator.is_local_main_process:
      transformers.utils.logging.set_verbosity_warning()
      diffusers.utils.logging.set_verbosity_info()
  else:
      transformers.utils.logging.set_verbosity_error()
      diffusers.utils.logging.set_verbosity_error()

  # Keep seed setting (Important for reproducibility)
  if args.seed is not None:
      set_seed(args.seed, device_specific=True)

  device = accelerator.device

  runner = build_model_runner(args)

  # For mixed precision training we cast all non-trainable weights to half-precision
  # as these weights are only used for inference, keeping weights in full precision is not required.
  weight_dtype = torch.float32
  if accelerator.mixed_precision == "fp16":
      weight_dtype = torch.float16
  elif accelerator.mixed_precision == "bf16":
      weight_dtype = torch.bfloat16

  if runner.vae:
      logger.info(f"Casting VAE to {weight_dtype}", main_process_only=False)
      runner.vae.to(dtype=weight_dtype)

  if runner.text_encoder:
      logger.info(f"Casting TextEncoder to {weight_dtype}", main_process_only=False)
      runner.text_encoder.to(dtype=weight_dtype)

  # building dataloader
  global_rank = accelerator.process_index
  anno_file = args.anno_file

  if args.task == 't2v':
    # For video generation training
    video_text_dataset = LengthGroupedVideoTextDataset(
        anno_file, 
        max_frames=args.max_frames,
        resolution=args.resolution,
        load_vae_latent=not args.load_vae,
        load_text_fea=not args.load_text_encoder,
    )

    
    train_dataloader = create_length_grouped_video_text_dataloader(
        video_text_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        epoch=args.seed,
    )


  
  logger.info("Building dataset finished")

  # building ema model
  model_ema = deepcopy(runner.dit) if args.ema_update else None
  if model_ema:
      model_ema.eval()

  # set the ema model not update by gradient
  if model_ema:
      model_ema.to(dtype=weight_dtype)
      for param in model_ema.parameters():
          param.requires_grad = False

  # report model details
  n_learnable_parameters = sum(p.numel() for p in runner.dit.parameters() if p.requires_grad)
  n_fix_parameters = sum(p.numel() for p in runner.dit.parameters() if not p.requires_grad)
  logger.info(f'total number of learnable params: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {n_learnable_parameters / 1e6} M')
  logger.info(f'total number of fixed params in : >>>>>>>>>>>>>>>>>>>>>>>> {n_fix_parameters / 1e6} M')

  print(f'total number of learnable params: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {n_learnable_parameters / 1e6} M')
  print(f'total number of fixed params in : >>>>>>>>>>>>>>>>>>>>>>>> {n_fix_parameters / 1e6} M')


  # `accelerate` 0.16.0 will have better support for customized saving
  # Register Hook to load and save model_ema
  # if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
  #     # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
  #     def save_model_hook(models, weights, output_dir):
  #         if accelerator.is_main_process:
  #             if model_ema:
  #                 model_ema_state = model_ema.state_dict()
  #                 torch.save(model_ema_state, os.path.join(output_dir, 'pytorch_model_ema.bin'))

  #     def load_model_hook(models, input_dir):
  #         if model_ema:
  #             model_ema_path = os.path.join(input_dir, 'pytorch_model_ema.bin')
  #             if os.path.exists(model_ema_path):
  #                 model_ema_state = torch.load(model_ema_path, map_location='cpu')
  #                 load_res = model_ema.load_state_dict(model_ema_state)
  #                 print(f"Loading ema weights {load_res}")

  #     accelerator.register_save_state_pre_hook(save_model_hook)
  #     accelerator.register_load_state_pre_hook(load_model_hook)


  # Create the Optimizer
  optimizer = create_optimizer(args, runner.dit)
  logger.info(f"optimizer: {optimizer}")


   # Create the LR scheduler
  num_training_steps_per_epoch = args.iters_per_epoch
  args.max_train_steps = args.epochs * num_training_steps_per_epoch
  warmup_iters = args.warmup_epochs * num_training_steps_per_epoch

  if args.warmup_steps > 0:
      warmup_iters = args.warmup_steps

  logger.info(f"LRScheduler: {args.lr_scheduler}, Warmup steps: {warmup_iters * args.gradient_accumulation_steps}")

  if args.lr_scheduler == 'cosine':
      lr_schedule_values = cosine_scheduler(
          args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
          warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
      ) 
  elif args.lr_scheduler == 'constant_with_warmup':
      lr_schedule_values = constant_scheduler(
          args.lr, args.epochs, num_training_steps_per_epoch, 
          warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
      )
  else:
      raise NotImplementedError(f"Not Implemented for scheduler {args.lr_scheduler}")


  # Wrap the model, optmizer, and scheduler with accelerate
  logger.info(f'before accelerator.prepare')

  # Only wrapping the trained dit and huge text encoder
  runner.dit, optimizer = accelerator.prepare(runner.dit, optimizer)


  # Load the VAE and EMAmodel to GPU
  if runner.vae:
      runner.vae.to(device)

  if runner.text_encoder:
      runner.text_encoder.to(device)

  logger.info(f'after accelerator.prepare')
  logger.info(f'{runner.dit}')


  if accelerator.is_main_process:
      accelerator.init_trackers(os.path.basename(args.output_dir), config=vars(args))

  # Report the training info
  total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
  logger.info("***** Running training *****")
  logger.info("LR = %.8f" % args.lr)
  logger.info("Min LR = %.8f" % args.min_lr)
  logger.info("Weigth Decay = %.8f" % args.weight_decay)
  logger.info("Batch size = %d" % total_batch_size)
  logger.info("Number of training steps = %d" % (num_training_steps_per_epoch * args.epochs))
  logger.info("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))


  # Report the training info
  total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
  print("***** Running training *****")
  print("LR = %.8f" % args.lr)
  print("Min LR = %.8f" % args.min_lr)
  print("Weigth Decay = %.8f" % args.weight_decay)
  print("Batch size = %d" % total_batch_size)
  print("Number of training steps = %d" % (num_training_steps_per_epoch * args.epochs))
  print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))


  # Auto resume the checkpoint
  initial_global_step = auto_resume(args, accelerator)
  first_epoch = initial_global_step // num_training_steps_per_epoch

  # Start Train!
  start_time = time.time()

  for epoch in range(first_epoch, args.epochs):
      train_stats = train_one_epoch_with_fsdp(
          runner,
          model_ema,
          accelerator,
          args.model_dtype,
          train_dataloader,
          optimizer,
          lr_schedule_values,
          device, 
          epoch, 
          args.clip_grad,
          start_steps=epoch * num_training_steps_per_epoch,
          args=args,
          print_freq=args.print_freq,
          iters_per_epoch=num_training_steps_per_epoch,
          ema_decay=args.ema_decay,
          use_temporal_pyramid=args.use_temporal_pyramid,
          
      )

      if args.output_dir:
          if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
              if accelerator.sync_gradients:
                  global_step = num_training_steps_per_epoch * (epoch + 1)
                  save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                  accelerator.save_state(save_path, safe_serialization=False)
                  logger.info(f"Saved state to {save_path}")

          

      log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                  'epoch': epoch, 'n_parameters': n_learnable_parameters}

      if args.output_dir:
          with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
              f.write(json.dumps(log_stats) + "\n")

  total_time = time.time() - start_time
  total_time_str = str(datetime.timedelta(seconds=int(total_time)))
  print('Training time {}'.format(total_time_str))

  accelerator.end_training()



  

  



if __name__ == "__main__":
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  opts = get_args()
  if opts.output_dir:
      Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
  main(opts)





