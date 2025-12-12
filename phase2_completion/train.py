# Copyright (c) 2025, Biao Zhang.

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

# Add root directory to sys.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Handle tensorboard import with fallback
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        SummaryWriter = None

import vecset.utils.misc as misc
from vecset.utils.objaverse import Objaverse
from vecset.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from vecset.models import autoencoder
from vecset.engines.engine_ae import train_one_epoch, evaluate

import wandb

def get_args_parser():
    parser = argparse.ArgumentParser('VecSetAutoEncoder', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='learnable_vec1024x16_dim1024_depth24', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--point_cloud_size', default=8192, type=int,
                        help='input size')
    parser.add_argument('--input_dim', default=13, type=int,
                        help='input dimension (3 for coords, more for extra features)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')


    # Dataset parameters
    parser.add_argument('--data_path', default='/home/zhanb0b/data/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=60, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')

    # Phase 2: Probabilistic Completion parameters
    parser.add_argument('--partial_prob', type=float, default=0.5,
                        help='Probability of masking input classes (default: 0.5)')
    parser.add_argument('--min_remove', type=int, default=1,
                        help='Minimum number of classes to remove (default: 1)')
    parser.add_argument('--max_remove', type=int, default=5,
                        help='Maximum number of classes to remove (default: 5)')

    return parser

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Initializing Dataset with path: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")

    # Verify CSV files existence (Objaverse internal check)
    import vecset.utils.objaverse as obj_utils
    csv_dir = os.path.dirname(obj_utils.__file__)
    for split in ['train', 'val']:
        csv_path = os.path.join(csv_dir, f'objaverse_{split}.csv')
        if not os.path.exists(csv_path):
             raise FileNotFoundError(f"Objaverse CSV not found: {csv_path}. Please run create_csv.py or sync files.")
    
    try:
        dataset_train = Objaverse(
            split='train', 
            sdf_sampling=True, 
            sdf_size=4096, 
            surface_sampling=True, 
            surface_size=args.point_cloud_size, 
            dataset_folder=args.data_path,
            partial_prob=args.partial_prob,
            min_remove=args.min_remove,
            max_remove=args.max_remove
        )
        dataset_val = Objaverse(split='val', sdf_sampling=True, sdf_size=4096, surface_sampling=True, surface_size=args.point_cloud_size, dataset_folder=args.data_path)
        print(f"Dataset initialized. Train: {len(dataset_train)}, Val: {len(dataset_val)}")
    except Exception as e:
        print(f"Fatal Error initializing Objaverse dataset: {e}")
        raise e

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        if SummaryWriter is not None:
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None
            print("Warning: Tensorboard not available, skipping logging.")
            
        if args.wandb:
            try:
                # Replace with your actual WandB API key
                print(f"Attempting WandB login...")
                wandb.login(key="d6891a1bb4397a24519ef1b36091aa1b77ea67e1")
                wandb.init(project="VecSetAutoEncoder", config=args)
                print(f"WandB initialized successfully.")
            except Exception as e:
                print(f"Warning: WandB initialization failed: {e}")
                print("Continuing without WandB logging...")
                args.wandb = False
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=2,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        prefetch_factor=2,
    )

    model = autoencoder.__dict__[args.model](pc_size=args.point_cloud_size, input_dim=args.input_dim)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # # build optimizer with layer-wise lr decay (lrd)
    # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #     no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #     layer_decay=args.layer_decay
    # )
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = torch.nn.L1Loss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_iou = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args
        )

        # Save 'last' checkpoint every epoch to ensure resume capability
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, tag='last')

        if epoch % 5 == 0 or epoch + 1 == args.epochs:
            print(f"Starting validation at epoch {epoch}...", flush=True)
            try:
                test_stats = evaluate(data_loader_val, model, device)
                print(f"Validation completed at epoch {epoch}.", flush=True)

                print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
                
                # Save 'best' checkpoint if IoU improves
                if test_stats["iou"] > max_iou:
                    max_iou = test_stats["iou"]
                    print(f'New max IoU: {max_iou:.2f}%. Saving best model...')
                    if args.output_dir:
                        misc.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, tag='best')
                else:
                    print(f'Max iou: {max_iou:.2f}%')

                if log_writer is not None:
                    # log_writer.add_scalar('perf/test_iou', test_stats['iou'], epoch)
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()},
                                'epoch': epoch,
                                'n_parameters': n_parameters}
            except Exception as e:
                print(f"Error during validation at epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                # Continue training even if validation fails
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                'epoch': epoch,
                                'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            if args.wandb:
                wandb.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)