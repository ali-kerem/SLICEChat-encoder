# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import random
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import wandb
import yaml

from .util import misc
from .util.misc import NativeScalerWithGradNormCount as NativeScaler
from .util.datasets import get_wsi_mae_dataset

from .factory import create_model_from_config

from .engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_config', default='src/slicechat_encoder/models/mamba/model_configs/base.yaml', type=str,
                        help='Path to Mamba MAE model config YAML file')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_ratio', type=float, default=0.1, metavar='N',
                        help='fraction of total epochs for LR warmup (default: 0.1 = 10%)')

    # Dataset parameters
    parser.add_argument('--data_path', default='/leonardo_work/EUHPC_B31_075/wsi-tensors/tcga/conch_v15', type=str,
                        help='dataset path containing features and coords subdirectories')
    parser.add_argument('--csv_path', default='/leonardo_work/EUHPC_B31_075/splits/train.csv', type=str,
                        help='csv path that contains the filenames of the features and coords')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Enable deterministic training (disables cudnn.benchmark, enables deterministic '
                             'algorithms). Slower and may break fused/flash kernels, but improves reproducibility.')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_interval', default=10, type=int,
                        help='save checkpoint every N epochs (default: 10)')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # Weights & Biases
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', default='mae-pretrain', type=str,
                        help='wandb project name')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='wandb run name (defaults to auto-generated)')

    return parser


def main(args):
    if args.distributed:
        misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.deterministic:
        # Trade throughput for reproducibility. Note: fused Mamba/flash-attention
        # kernels may not support deterministic algorithms and could error here.
        cudnn.benchmark = False
        cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.benchmark = True

    # Create dataloader for WSI features
    data_loader_train = get_wsi_mae_dataset(args, is_train=True)
    print(f"Dataset: {len(data_loader_train.dataset)} samples")

    if misc.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    if args.wandb and misc.is_main_process():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        print(f"Initialized wandb: project={args.wandb_project}, run={wandb.run.name}")

    # Create model
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    model = create_model_from_config(model_config)
    print(f"Created model: {type(model).__name__}")
    # Save model config to output dir for reproducibility
    if args.output_dir and misc.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'model_cfg.yaml'), 'w') as f:
            yaml.dump(model_config, f)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model = {type(model).__name__}")
    print(f"Number of trainable params: {n_parameters / 1e6:.2f}M")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    args.warmup_epochs = int(args.warmup_ratio * args.epochs)

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("warmup epochs: %d (%.0f%% of %d)" % (args.warmup_epochs, args.warmup_ratio * 100, args.epochs))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(data_loader_train, 'sampler') and data_loader_train.sampler is not None:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_interval == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if args.wandb and misc.is_main_process():
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
