# Copyright (c) 2025, Biao Zhang.

import math
import sys
from typing import Iterable

from numpy import inner
import torch
import torch.nn.functional as F

import utils.misc as misc
import utils.lr_sched as lr_sched


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
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.L1Loss()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.logdir))

    for data_iter_step, (points, labels, surface, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).squeeze(-1)
        surface = surface.to(device, non_blocking=True)
        # surface_normals = surface_normals.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            points = points.requires_grad_(True)
            points_all = torch.cat([points, surface[:, :, :3]], dim=1)
            outputs = model(surface, points_all)

            output = outputs['o']
            
            # output: (B, N, 11) -> SDF (0), Labels (1..10)
            # labels: (B, N, 2) -> SDF (0), LabelIdx (1)
            
            pred_sdf = output[:, :, 0]
            pred_logits = output[:, :, 1:]
            
            target_sdf = labels[:, :, 0]
            target_labels_idx = labels[:, :, 1].long()
            
            grad = points_gradient(points_all, output[:, :, 0:1]) # Grad of SDF channel only

            loss_eikonal = (grad[:, :1024].norm(2, dim=-1) - 1).pow(2).mean() # Eikonal on Vol pts?
            # Note: Vol points are first 1024?
            # Objaverse returns cat([vol, near]). Vol is first.
            
            loss_vol = criterion(pred_sdf[:, :1024], target_sdf[:, :1024])
            loss_near = criterion(pred_sdf[:, 1024:2048], target_sdf[:, 1024:2048])
            loss_surface = (pred_sdf[:, 2048:]).abs().mean()
            
            # Classification Loss
            # Background class (0) implicitly handled?
            # If using BCEWithLogitsLoss(pred_logits, target_one_hot):
            # We have 10 logits for classes 1-10.
            # If label=0, target_one_hot is all zeros (good).
            # If label=k, target_one_hot[k-1] = 1.
            
            target_one_hot = F.one_hot(target_labels_idx, num_classes=11)[:, :, 1:].float()
            loss_cls = F.binary_cross_entropy_with_logits(pred_logits, target_one_hot)

            loss = loss_vol + 50 * loss_near + 0.001 * loss_eikonal + 100 * loss_surface + 1.0 * loss_cls


        loss_value = loss.item()

        threshold = 0

        vol_iou = calc_iou(pred_sdf[:, :1024], target_sdf[:, :1024], threshold)
        near_iou = calc_iou(pred_sdf[:, 1024:2048], target_sdf[:, 1024:2048], threshold)
        
        # Calculate Classification Accuracy
        # Pred Class = argmax(logits) + 1? Or threshold?
        # If max(logits) < 0, class 0?
        # Let's use argmax over (0 vector, logits).
        pred_probs = torch.cat([torch.zeros_like(pred_logits[:, :, :1]), pred_logits], dim=2)
        pred_cls = torch.argmax(pred_probs, dim=2)
        acc = (pred_cls == target_labels_idx).float().mean()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        metric_logger.update(loss_vol=loss_vol.item())
        metric_logger.update(loss_near=loss_near.item())
        metric_logger.update(loss_eikonal=loss_eikonal.item())
        metric_logger.update(loss_surface=loss_surface.item())
        # metric_logger.update(loss_surface_normal=loss_surface_normal.item())

        metric_logger.update(vol_iou=vol_iou.item())
        metric_logger.update(near_iou=near_iou.item())
        metric_logger.update(acc=acc.item())

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
def evaluate(data_loader, model, device):
    criterion = torch.nn.L1Loss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        points = batch[0]
        labels = batch[1]
        surface = batch[2]
        
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).squeeze(-1)
        surface = surface.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            points_all = torch.cat([points, surface[:, :, :3]], dim=1)
            outputs = model(surface, points_all)
            output = outputs['o']
            
            # output: (B, N, 11)
            pred_sdf = output[:, :, 0]
            pred_logits = output[:, :, 1:]
            
            target_sdf = labels[:, :, 0]
            target_labels_idx = labels[:, :, 1].long()
            
            # Validation logic
            # Objaverse returns 2048 points (Vol+Near)
            
            loss_vol = criterion(pred_sdf[:, :1024], target_sdf[:, :1024])
            loss_near = criterion(pred_sdf[:, 1024:2048], target_sdf[:, 1024:2048])
            loss_surface = (pred_sdf[:, 2048:]).abs().mean()
            
            target_one_hot = F.one_hot(target_labels_idx, num_classes=11)[:, :, 1:].float()
            loss_cls = F.binary_cross_entropy_with_logits(pred_logits, target_one_hot)
            
            loss = loss_vol + 10 * loss_near + 10 * loss_surface + loss_cls

            pred_probs = torch.cat([torch.zeros_like(pred_logits[:, :, :1]), pred_logits], dim=2)
            pred_cls = torch.argmax(pred_probs, dim=2)
            acc = (pred_cls == target_labels_idx).float().mean()
            
            vol_iou = calc_iou(pred_sdf[:, :1024], target_sdf[:, :1024], 0)
            near_iou = calc_iou(pred_sdf[:, 1024:2048], target_sdf[:, 1024:2048], 0)

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_vol=loss_vol.item())
        metric_logger.update(loss_near=loss_near.item())
        metric_logger.update(loss_surface=loss_surface.item())
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.update(vol_iou=vol_iou.item())
        metric_logger.update(near_iou=near_iou.item())
        metric_logger.update(acc=acc.item())
        
        # Combined IoU for reporting
        metric_logger.update(iou=(vol_iou.item() + near_iou.item()) / 2.0)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
