# Copyright (c) 2025, Biao Zhang.
# Modified for Phase 1: Multi-class (10 classes) training

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import sys
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vecset.utils import misc
from vecset.utils.cardiac_dataset import CardiacPerClass
from vecset.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from vecset.models import autoencoder
from vecset.engines.engine_ae import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('VecSet Phase 1 - Multi-class Reconstruction', add_help=False)
    
    # Training parameters
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Gradient accumulation iterations')

    # Model parameters
    parser.add_argument('--model', default='learnable_vec1024x16_dim1024_depth24_nb', type=str,
                        help='Model architecture')
    parser.add_argument('--point_cloud_size', default=8192, type=int,
                        help='Number of surface points')
    parser.add_argument('--sdf_size', default=4096, type=int,
                        help='Number of SDF query points')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Gradient clipping norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5,
                        help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Warmup epochs')

    # Dataset parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')
    # Phase 1: All 10 classes
    parser.add_argument('--classes', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help='Class IDs to train on (default: all 10 classes)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')

    # Output parameters
    parser.add_argument('--output_dir', default='./output/',
                        help='Output directory')
    parser.add_argument('--log_dir', default='./output/',
                        help='Tensorboard log directory')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Device parameters
    parser.add_argument('--device', default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--dist_on_itp', action='store_true',
                        help='Enable distributed training on ITP cluster')

    # WandB parameters
    parser.add_argument('--wandb_project', default='vecset-phase1', type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--use_wandb', action='store_true')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('=' * 60)
    print('VecSet Phase 1: Multi-class Reconstruction (10 Classes)')
    print('=' * 60)
    print(f"Job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"Args:\n{args}".replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Enable TF32 for faster training on A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Create datasets
    print(f"\nLoading data from: {args.data_path}")
    print(f"Classes: {args.classes}")
    
    dataset_train = CardiacPerClass(
        dataset_folder=args.data_path,
        split='train',
        classes=args.classes,
        surface_size=args.point_cloud_size,
        sdf_size=args.sdf_size,
        augment=True,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    
    dataset_val = CardiacPerClass(
        dataset_folder=args.data_path,
        split='val',
        classes=args.classes,
        surface_size=args.point_cloud_size,
        sdf_size=args.sdf_size,
        augment=False,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Create samplers for distributed training
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    print(f"Sampler_train = {sampler_train}")

    # Setup logging
    global_rank = misc.get_rank()
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Create data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=2,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Create model
    model = autoencoder.__dict__[args.model](
        pc_size=args.point_cloud_size,
        input_dim=3,
        output_dim=1,
    )
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel: {args.model}")
    print(f"Number of params: {n_parameters / 1e6:.2f}M")

    # Calculate effective batch size
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"\nBase LR: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual LR: {args.lr:.2e}")
    print(f"Accumulate grad iterations: {args.accum_iter}")
    print(f"Effective batch size: {eff_batch_size}")

    # Wrap model for distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    # Create optimizer and loss
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()
    criterion = torch.nn.L1Loss()

    print(f"Criterion: {criterion}")

    # Load checkpoint if resuming
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Initialize WandB
    if args.use_wandb and global_rank == 0:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"phase1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
        )

    # Training loop
    print(f"\nStart training for {args.epochs} epochs")
    start_time = time.time()
    best_iou = 0.0  # Track best validation IoU
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # Train one epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args
        )

        # Validation every 10 epochs or last epoch
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            val_stats = evaluate(model, criterion, data_loader_val, device, args)
            current_iou = (val_stats.get('vol_iou', 0.0) + val_stats.get('near_iou', 0.0)) / 2
            
            # Print validation stats (rank 0 only usually, but misc.all_reduce might handle sync)
            # Evaluate usually returns stats only on rank 0 or syncs them? 
            # In misc.evaluate it usually syncs. 
            # We strictly print on rank 0 just in case.
            if misc.is_main_process():
                print(f"Validation IoU: {val_stats.get('vol_iou', 0.0):.4f} (vol), {val_stats.get('near_iou', 0.0):.4f} (near), avg: {current_iou:.4f}")
            
                # Save checkpoint if this is the best model so far
                if current_iou > best_iou:
                    print(f"New best IoU: {current_iou:.4f} (previous: {best_iou:.4f})")
                    best_iou = current_iou
                    
                    if args.output_dir:
                        # Delete old best checkpoint
                        old_best = os.path.join(args.output_dir, 'checkpoint-best.pth')
                        if os.path.exists(old_best):
                            os.remove(old_best)
                            print(f"Removed old checkpoint: {old_best}")
                        
                        # Save new best checkpoint
                        misc.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, tag='best'
                        )
                        print(f"Saved best checkpoint with IoU: {current_iou:.4f}")
            
            # Broadcast best_iou to other ranks to keep them in sync if needed (though not strictly necessary if training proceeds)
            # But let's just make sure log_stats is consistent
            
            # Log validation stats (Rank 0 only handles logging)
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,
                'best_iou': best_iou
            }
            
            # Log to tensorboard
            if log_writer is not None:
                for k, v in val_stats.items():
                    log_writer.add_scalar(f'val/{k}', v, epoch)
                log_writer.add_scalar('val/best_iou', best_iou, epoch)
        else:
            # Log only training stats
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # Log to WandB
        if args.use_wandb and global_rank == 0:
            wandb.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\nTraining completed in {total_time_str}')

    if args.use_wandb and global_rank == 0:
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
