# Copyright (c) 2025, Biao Zhang.

import math
import sys
from typing import Iterable

from numpy import inner
import torch
import torch.nn.functional as F

from vecset.utils import misc
from vecset.utils import lr_sched


def calc_iou(output, labels, threshold):
    target = torch.zeros_like(labels)
    target[labels>=threshold] = 1
    
    pred = torch.zeros_like(output)
    pred[output>=threshold] = 1

    accuracy = (pred==target).float().sum(dim=1) / target.shape[1]
    accuracy = accuracy.mean()
    intersection = (pred * target).sum(dim=1)
    union = (pred + target).gt(0).sum(dim=1) + 1e-5
    iou = intersection * 1.0 / union
    iou = iou.mean()
    return iou

def calc_dice(output, labels, threshold):
    """Calculate Dice coefficient (F1 score)"""
    target = torch.zeros_like(labels)
    target[labels>=threshold] = 1
    
    pred = torch.zeros_like(output)
    pred[output>=threshold] = 1
    
    intersection = (pred * target).sum(dim=1)
    dice = (2.0 * intersection) / (pred.sum(dim=1) + target.sum(dim=1) + 1e-5)
    dice = dice.mean()
    return dice

def points_gradient(inputs, outputs):
    d_points = torch.ones_like(
        outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # Custom formats for different metrics
    metric_logger.add_meter('vol_iou', misc.SmoothedValue(fmt='({global_avg:.4f})'))  # Only avg
    metric_logger.add_meter('near_iou', misc.SmoothedValue(fmt='({global_avg:.4f})'))  # Only avg
    metric_logger.add_meter('vol_dice', misc.SmoothedValue(fmt='{value:.4f}'))  # Only current
    metric_logger.add_meter('near_dice', misc.SmoothedValue(fmt='{value:.4f}'))  # Only current
    metric_logger.add_meter('loss', misc.SmoothedValue(fmt='{value:.4f}'))  # Only current
    metric_logger.add_meter('grad_norm', misc.SmoothedValue(fmt='{value:.2e}'))  # Gradient norm
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.L1Loss()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (points, labels, surface, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)
        # surface_normals = surface_normals.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            points = points.requires_grad_(True)
            points_all = torch.cat([points, surface], dim=1)
            outputs = model(surface, points_all)

            output = outputs['o']

            grad = points_gradient(points_all, output)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                # Use sdf_size from args (default 4096) instead of hardcoded 1024
                sdf_size = args.sdf_size  # 4096: vol=0:4096, near=4096:8192, surface=8192+
                loss_eikonal = (grad[:, :].norm(2, dim=-1) - 1).pow(2).mean()
                loss_vol = criterion(output[:, :sdf_size], labels[:, :sdf_size])
                loss_near = criterion(output[:, sdf_size:2*sdf_size], labels[:, sdf_size:2*sdf_size])
                loss_surface = (output[:, 2*sdf_size:]).abs().mean()
                
                # print(grad.shape, surface_normals.shape)
                # inner = torch.einsum('b n c, b n c -> b n', grad[:, 2048:], surface_normals)
                
                # print(inner.max(), inner.min(), inner.mean())
                # print(F.l1_loss(grad[:, 2048:], surface_normals), F.l1_loss(grad[:, 2048:], -surface_normals))
                # loss_surface_normal = F.l1_loss(F.normalize(grad[:, 2048:], dim=2), surface_normals)
                # loss_surface_normal = 1 - torch.einsum('b n c, b n c -> b n', (F.normalize(grad[:, 2048:], dim=2, eps=1e-6), surface_normals)).mean()

                # Reduced Eikonal weight to prevent explosion: 0.001 -> 0.0001
                loss = loss_vol + 10 * loss_near + 0.001 * loss_eikonal + 1 * loss_surface# + 0.01 * loss_surface_normal


        loss_value = loss.item()

        threshold = 0

        sdf_size = args.sdf_size  # Use consistent indexing
        vol_iou = calc_iou(output[:, :sdf_size], labels[:, :sdf_size], threshold)
        near_iou = calc_iou(output[:, sdf_size:2*sdf_size], labels[:, sdf_size:2*sdf_size], threshold)
        vol_dice = calc_dice(output[:, :sdf_size], labels[:, :sdf_size], threshold)
        near_dice = calc_dice(output[:, sdf_size:2*sdf_size], labels[:, sdf_size:2*sdf_size], threshold)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        # Calculate gradient norm for monitoring
        if (data_iter_step + 1) % accum_iter == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            metric_logger.update(grad_norm=total_norm)
            
            optimizer.zero_grad()
        
        torch.cuda.synchronize()

        # Only log essential metrics
        metric_logger.update(loss=loss_value)
        metric_logger.update(vol_iou=vol_iou.item())
        metric_logger.update(near_iou=near_iou.item())
        metric_logger.update(vol_dice=vol_dice.item())
        metric_logger.update(near_dice=near_dice.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module,
             data_loader: Iterable, device: torch.device, args=None):
    """
    Evaluate model on validation set without gradient computation.
    """
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Val:'
    
    for data_iter_step, (points, labels, surface, _, _) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            # Temporarily enable gradients for Eikonal loss computation
            with torch.enable_grad():
                points = points.requires_grad_(True)
                points_all = torch.cat([points, surface], dim=1)
                outputs = model(surface, points_all)
                output = outputs['o']
                
                grad = points_gradient(points_all, output)
                
                # Compute losses
                loss_eikonal = (grad[:, :].norm(2, dim=-1) - 1).pow(2).mean()
            
            # Compute other losses without gradients - use sdf_size from args
            sdf_size = args.sdf_size  # 4096: vol=0:4096, near=4096:8192, surface=8192+
            loss_vol = criterion(output[:, :sdf_size], labels[:, :sdf_size])
            loss_near = criterion(output[:, sdf_size:2*sdf_size], labels[:, sdf_size:2*sdf_size])
            loss_surface = (output[:, 2*sdf_size:]).abs().mean()
            loss = loss_vol + 10 * loss_near + 0.001 * loss_eikonal + 1 * loss_surface
        
        # Compute IoU and Dice - use sdf_size from args
        threshold = 0
        vol_iou = calc_iou(output[:, :sdf_size], labels[:, :sdf_size], threshold)
        near_iou = calc_iou(output[:, sdf_size:2*sdf_size], labels[:, sdf_size:2*sdf_size], threshold)
        vol_dice = calc_dice(output[:, :sdf_size], labels[:, :sdf_size], threshold)
        near_dice = calc_dice(output[:, sdf_size:2*sdf_size], labels[:, sdf_size:2*sdf_size], threshold)
        
        # Update metrics
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_vol=loss_vol.item())
        metric_logger.update(loss_near=loss_near.item())
        metric_logger.update(loss_eikonal=loss_eikonal.item())
        metric_logger.update(loss_surface=loss_surface.item())
        metric_logger.update(vol_iou=vol_iou.item())
        metric_logger.update(near_iou=near_iou.item())
        metric_logger.update(vol_dice=vol_dice.item())
        metric_logger.update(near_dice=near_dice.item())
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Validation stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

