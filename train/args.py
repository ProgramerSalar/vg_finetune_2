
import argparse



def get_args():
    parser = argparse.ArgumentParser('Pyramid-Flow Multi-process Training script', add_help=False)
    parser.add_argument('--task', default='t2v', type=str, choices=["t2v", "t2i"], help="Training image generation or video generation")
    parser.add_argument('--batch_size', default=4, type=int, help="The per device batch size")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--iters_per_epoch', default=2000, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Model parameters
    parser.add_argument('--ema_update', action='store_true')
    parser.add_argument('--ema_decay', default=0.9999, type=float, metavar='MODEL', help='ema decay rate')
    parser.add_argument('--load_ema_model', default='', type=str, help='The ema model checkpoint loading')
    parser.add_argument('--model_name', default='pyramid_flux', type=str, help="The Model Architecture Name", choices=["pyramid_flux", "pyramid_mmdit"])
    parser.add_argument('--model_path', default='', type=str, help='The pre-trained dit weight path')
    parser.add_argument('--model_variant', default='diffusion_transformer_384p', type=str, help='The dit model variant', choices=['diffusion_transformer_768p', 'diffusion_transformer_384p', 'diffusion_transformer_image'])
    parser.add_argument('--model_dtype', default='bf16', type=str, help="The Model Dtype: bf16 or fp16", choices=['bf16', 'fp16'])
    parser.add_argument('--load_model_ema_to_cpu', action='store_true')

    # FSDP condig
    parser.add_argument('--use_fsdp', action='store_true')
    parser.add_argument('--fsdp_shard_strategy', default='zero2', type=str, choices=['zero2', 'zero3'])

    # The training manner config
    parser.add_argument('--use_flash_attn', action='store_true')
    parser.add_argument('--use_temporal_causal', action='store_true', default=True)
    parser.add_argument('--interp_condition_pos', action='store_true', default=True)
    parser.add_argument('--sync_video_input', action='store_true', help="whether to sync the video input")
    parser.add_argument('--load_text_encoder', action='store_true', help="whether to load the text encoder during training")
    parser.add_argument('--load_vae', action='store_true', help="whether to load the video vae during training")

    # Sequence Parallel config
    parser.add_argument('--use_sequence_parallel', action='store_true')
    parser.add_argument('--sp_group_size', default=1, type=int, help="The group size of sequence parallel")
    parser.add_argument('--sp_proc_num', default=-1, type=int, help="The number of process used for video training, default=-1 means using all process. This args indicated using how many processes for video training")

    # Model input config
    parser.add_argument('--max_frames', default=16, type=int, help='number of max video frames')
    parser.add_argument('--frame_per_unit', default=1, type=int, help="The number of frames per training unit")
    parser.add_argument('--schedule_shift', default=1.0, type=float, help="The flow matching schedule shift")
    parser.add_argument('--corrupt_ratio', default=1/3, type=float, help="The corruption ratio for the clean history in AR training")

    # Dataset Cconfig
    parser.add_argument('--anno_file', default='', type=str, help="The annotation jsonl file")
    parser.add_argument('--resolution', default='384p', type=str, help="The input resolution", choices=['384p', '768p'])

    # Training set config
    parser.add_argument('--dit_pretrained_weight', default='', type=str, help='The pretrained dit checkpoint')  
    parser.add_argument('--vae_pretrained_weight', default='', type=str,)
    parser.add_argument('--not_add_normalize', action='store_true')
    parser.add_argument('--use_temporal_pyramid', action='store_true', help="Whether to use the AR temporal pyramid training for video generation")
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_checkpointing_ratio', type=float, default=0.75, help="The ratio of transformer blocks used for gradient_checkpointing")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--video_sync_group', default=8, type=int, help="The number of process that accepts the same input video, used for temporal pyramid AR training. \
        This contributes to stable AR training. We recommend to set this value to 4, 8 or 16. If you have enough GPUs, set it equals to max_frames (16 for 5s, 32 for 10s), \
            make sure to satisfy `max_frames % video_sync_group == 0`")

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_beta1', default=0.9, type=float, metavar='BETA1',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--opt_beta2', default=0.999, type=float, metavar='BETA2',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Dataset parameters
    parser.add_argument('--output_dir', type=str, default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--logging_dir', type=str, default='log', help='path where to tensorboard log')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    # Distributed Training parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--global_step', default=0, type=int, metavar='N', help='The global optimization step')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training', type=str)

    return parser.parse_args()