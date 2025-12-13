# Copyright (c) 2025, Biao Zhang.

import math
import sys
from typing import Iterable

from numpy import inner
import torch
import torch.nn.functional as F

import vecset.utils.misc as misc
import vecset.utils.lr_sched as lr_sched


def calc_iou(output, labels, threshold):
    target = torch.zeros_like(labels)
    target[labels<threshold] = 1
    
    pred = torch.zeros_like(output)
    pred[output<threshold] = 1

    accuracy = (pred==target).float().sum(dim=1) / target.shape[1]
    accuracy = accuracy.mean()
    intersection = (pred * target).sum(dim=1)
    union = (pred + target).gt(0).sum(dim=1) + 1e-5
    iou = intersection * 1.0 / union
    iou = iou.mean()
    return iou

def calc_dice(output, labels, threshold):
    target = torch.zeros_like(labels)
    target[labels<threshold] = 1
    
    pred = torch.zeros_like(output)
    pred[output<threshold] = 1

    intersection = (pred * target).sum(dim=1)
    # Dice = 2*Inter / (|A| + |B|)
    # |A| + |B| = (pred.sum + target.sum)
    cardinality = pred.sum(dim=1) + target.sum(dim=1) + 1e-5
    dice = (2. * intersection) / cardinality
    return dice.mean()

# ============================================================================
# Advanced Loss Functions for Small Structure Improvement
# ============================================================================

# Per-class SDF loss weights (1-indexed: Myo, LA, LV, RA, RV, Ao, PA, LAA, Cor, PV)
# Higher weights for small/difficult structures: Aorta(6), PA(7), Coronary(9), PV(10)
CLASS_SDF_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0]

def focal_bce_loss(pred, target, gamma=2.0, alpha=0.25, reduction='mean'):
    """
    Focal Loss for multi-label binary classification.
    Focuses learning on hard-to-classify examples.
    
    Args:
        pred: (B, N, C) logits
        target: (B, N, C) one-hot targets
        gamma: focusing parameter (higher = more focus on hard examples)
        alpha: class balance weight
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)  # pt = p if y=1, 1-p if y=0
    focal_weight = alpha * (1 - pt) ** gamma
    focal_loss = focal_weight * bce
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'none':
        return focal_loss
    else:
        return focal_loss.sum()

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

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        points = batch[0]
        labels = batch[1]
        surface = batch[2]
        
        valid_class_mask = None
        if len(batch) > 5:
            valid_class_mask = batch[5].to(device, non_blocking=True)

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
            
            # Eikonal only on Volume points (Pos + Neg = 2048)
            loss_eikonal = (grad[:, :2048].norm(2, dim=-1) - 1).pow(2).mean() # Eikonal on Vol pts?
            # Note: Vol points are first 1024?
            # Objaverse returns cat([vol, near]). Vol is first.
            
            loss_vol = criterion(pred_sdf[:, :2048], target_sdf[:, :2048])
            loss_near = criterion(pred_sdf[:, 2048:3072], target_sdf[:, 2048:3072])
            loss_surface = (pred_sdf[:, 3072:]).abs().mean()
            
            # ================================================================
            # Per-Class Weighted SDF Loss (for Vol and Near points)
            # ================================================================
            # Get class weights tensor
            class_weights_tensor = torch.tensor(CLASS_SDF_WEIGHTS, device=device, dtype=torch.float32)
            
            # Get per-point weights based on ground truth label (clamped to valid range)
            # target_labels_idx: (B, N) with values 0-10 (0=background)
            # For background (0), use weight 1.0
            point_labels_clamped = target_labels_idx.clamp(0, len(CLASS_SDF_WEIGHTS))
            # Create weight lookup with 1.0 for background at index 0
            weights_with_bg = torch.cat([torch.ones(1, device=device), class_weights_tensor])
            point_weights = weights_with_bg[point_labels_clamped]  # (B, N)
            
            # Compute weighted SDF losses
            loss_vol_raw = F.l1_loss(pred_sdf[:, :2048], target_sdf[:, :2048], reduction='none')
            loss_vol_weighted = (loss_vol_raw * point_weights[:, :2048]).mean()
            
            loss_near_raw = F.l1_loss(pred_sdf[:, 2048:3072], target_sdf[:, 2048:3072], reduction='none')
            loss_near_weighted = (loss_near_raw * point_weights[:, 2048:3072]).mean()
            
            # Classification Loss
            # Background class (0) implicitly handled?
            # If using BCEWithLogitsLoss(pred_logits, target_one_hot):
            # We have K logits for classes 1-K.
            # If label=0, target_one_hot is all zeros (good).
            # If label=k, target_one_hot[k-1] = 1.
            
            num_cls = getattr(args, 'nb_classes', 10)
            
            target_one_hot_query = F.one_hot(target_labels_idx, num_classes=num_cls+1)[:, :, 1:].float()
            # Concatenate surface labels (which are already one-hot in dims 3:3+num_cls)
            # Surface shape is (B, 8192, 3+num_cls)
            target_one_hot_surface = surface[:, :, 3:]
            
            target_one_hot = torch.cat([target_one_hot_query, target_one_hot_surface], dim=1)
            
            # ================================================================
            # Focal Loss for Classification (focuses on hard examples)
            # ================================================================
            if valid_class_mask is not None:
                # valid_class_mask: (B, num_cls) -> Expand to (B, N, num_cls)
                # pred_logits: (B, N, num_cls)
                mask_expanded = valid_class_mask.unsqueeze(1).expand(-1, pred_logits.shape[1], -1)
                
                loss_cls_raw = focal_bce_loss(pred_logits, target_one_hot, gamma=2.0, alpha=0.25, reduction='none')
                loss_cls = (loss_cls_raw * mask_expanded).sum() / (mask_expanded.sum() + 1e-5)
            else:
                loss_cls = focal_bce_loss(pred_logits, target_one_hot, gamma=2.0, alpha=0.25)

            # Use weighted losses instead of original
            loss = loss_vol_weighted + 0.5 * loss_near_weighted + 0.001 * loss_eikonal + 0.5 * loss_surface + 5.0 * loss_cls


        loss_value = loss.item()

        threshold = 0

        vol_iou = calc_iou(pred_sdf[:, :2048], target_sdf[:, :2048], threshold)
        near_iou = calc_iou(pred_sdf[:, 2048:3072], target_sdf[:, 2048:3072], threshold)
        
        # Calculate Classification Accuracy
        # Pred Class = argmax(logits) + 1? Or threshold?
        # If max(logits) < 0, class 0?
        # Let's use argmax over (0 vector, logits).
        pred_probs = torch.cat([torch.zeros_like(pred_logits[:, :, :1]), pred_logits], dim=2)
        pred_cls = torch.argmax(pred_probs, dim=2)
        # Only compute accuracy on query points as target_labels_idx is for query only
        acc = (pred_cls[:, :3072] == target_labels_idx).float().mean()

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
        
        # Also log Dice for training stability monitoring
        vol_dice = calc_dice(pred_sdf[:, :2048], target_sdf[:, :2048], threshold)
        near_dice = calc_dice(pred_sdf[:, 2048:3072], target_sdf[:, 2048:3072], threshold)
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
def evaluate(data_loader, model, device):
    criterion = torch.nn.L1Loss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    # We need args here, but signature is fixed? 
    # Usually evaluate calls take args, or we can infer from model output shape?
    # Model output shape is (B, N, 1 + nb_classes).
    # We can infer nb_classes = output.shape[-1] - 1.

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
            
            nb_classes = output.shape[-1] - 1
            
            # output: (B, N, 1 + nb_classes)
            pred_sdf = output[:, :, 0]
            pred_logits = output[:, :, 1:]
            
            target_sdf = labels[:, :, 0]
            target_labels_idx = labels[:, :, 1].long()
            
            # Calculate classification predictions FIRST
            pred_probs = torch.cat([torch.zeros_like(pred_logits[:, :, :1]), pred_logits], dim=2)
            pred_cls = torch.argmax(pred_probs, dim=2)

            # Validation logic
            num_points = points.shape[1]
            
            if num_points == 1024:
                # Validation set (near points only)
                loss_vol = torch.tensor(0.0, device=device)
                loss_near = criterion(pred_sdf, target_sdf)
                loss_surface = (pred_sdf[:, 1024:]).abs().mean() if pred_sdf.shape[1] > 1024 else torch.tensor(0.0, device=device) 
                
                vol_iou = torch.tensor(0.0, device=device)
                near_iou = calc_iou(pred_sdf[:, :1024], target_sdf[:, :1024], 0)
                
                vol_dice = torch.tensor(0.0, device=device)
                near_dice = calc_dice(pred_sdf[:, :1024], target_sdf[:, :1024], 0)
                 
                acc_points = pred_cls[:, :1024]
                acc_targets = target_labels_idx[:, :1024]
                 
            else: # Standard Mode (3072 or other)
                 # Default fallback to slicing assumption if 3072
                 # If not 3072, this might still range error if < 3072. 
                 # But let's assume if not 1024, it's the full training-like set.
                 loss_vol = criterion(pred_sdf[:, :2048], target_sdf[:, :2048])
                 loss_near = criterion(pred_sdf[:, 2048:3072], target_sdf[:, 2048:3072])
                 loss_surface = (pred_sdf[:, 3072:]).abs().mean() if pred_sdf.shape[1] > 3072 else torch.tensor(0.0, device=device)
                 
                 vol_iou = calc_iou(pred_sdf[:, :2048], target_sdf[:, :2048], 0)
                 near_iou = calc_iou(pred_sdf[:, 2048:3072], target_sdf[:, 2048:3072], 0)

                 vol_dice = calc_dice(pred_sdf[:, :2048], target_sdf[:, :2048], 0)
                 near_dice = calc_dice(pred_sdf[:, 2048:3072], target_sdf[:, 2048:3072], 0)

                 acc_points = pred_cls[:, :3072]
                 acc_targets = target_labels_idx[:, :3072]
            
            # --- Per-Class IoU Calculation ---
            eval_limit = 1024 if num_points == 1024 else 3072
            # Ensure we don't go out of bounds if shape is smaller (e.g. debugging)
            eval_limit = min(eval_limit, pred_sdf.shape[1])
            
            p_full = pred_sdf[:, :eval_limit]
            t_full = target_sdf[:, :eval_limit]
            l_full = target_labels_idx[:, :eval_limit]

            for c in range(1, nb_classes + 1):
                class_mask = (l_full == c)
                if class_mask.sum() > 0:
                    p_c = p_full[class_mask]
                    t_c = t_full[class_mask]
                    
                    pred_bin = (p_c < 0).float()
                    target_bin = (t_c < 0).float()
                    
                    # IoU
                    inter = (pred_bin * target_bin).sum()
                    union = (pred_bin + target_bin).gt(0).float().sum() + 1e-5
                    iou_c = inter / union
                    metric_logger.update(**{f'iou_class_{c}': iou_c.item()})
                    
                    # Dice 
                    # 2*Inter / (A+B)
                    card_c = pred_bin.sum() + target_bin.sum() + 1e-5
                    dice_c = (2. * inter) / card_c
                    metric_logger.update(**{f'dice_class_{c}': dice_c.item()})
            # ---------------------------------------

            acc = (acc_points == acc_targets).float().mean()
            # Classification loss logic (optional for eval but good for tracking)
            # We don't have surface labels here easily matched without re-slicing logic matching train.
            # Just skip loss_cls for simplicty in eval or set to 0
            loss_cls = torch.tensor(0.0, device=device)
            
            loss = loss_vol + 1.0 * loss_near + 1.0 * loss_surface + loss_cls

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_vol=loss_vol.item())
        metric_logger.update(loss_near=loss_near.item())
        metric_logger.update(loss_surface=loss_surface.item())
        metric_logger.update(vol_iou=vol_iou.item())
        metric_logger.update(near_iou=near_iou.item())
        metric_logger.update(vol_dice=vol_dice.item())
        metric_logger.update(near_dice=near_dice.item())
        metric_logger.update(acc=acc.item())
        
        # Combined IoU for reporting
        metric_logger.update(iou=(vol_iou.item() + near_iou.item()) / 2.0)
        metric_logger.update(dice=(vol_dice.item() + near_dice.item()) / 2.0)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
