# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import wandb

from .util import misc, lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    """
    Train for one epoch. Supports both:
    - Original MAE with images: dataloader yields (images, labels) tuples
    - LongNet MAE with WSI features: dataloader yields {"features", "coords", "padding_masks"} dicts
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Handle both data formats
        if isinstance(batch, dict):
            # WSI feature format: {"features", "coords", "padding_masks"}
            features = batch["features"].to(device, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            padding_masks = batch["padding_masks"].to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device.type):
                loss, _, _ = model(features, coords, padding_masks, mask_ratio=args.mask_ratio)
        else:
            # Original image format: (images, labels) tuple
            samples, _ = batch
            samples = samples.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        # Update grad norm if available
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            if grad_norm is not None:
                log_writer.add_scalar('grad_norm', grad_norm, epoch_1000x)
        
        if args.wandb and misc.is_main_process() and (data_iter_step + 1) % accum_iter == 0:
            wandb_log_dict = {
                'train/loss': loss_value_reduce,
                'train/lr': lr,
                'train/epoch': epoch + data_iter_step / len(data_loader),
            }
            if grad_norm is not None:
                wandb_log_dict['train/grad_norm'] = grad_norm
            wandb.log(wandb_log_dict)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}