# Copyright (c) 2025, Biao Zhang.
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_train = Objaverse(split='train', sdf_sampling=True, sdf_size=1024, surface_sampling=True, surface_size=args.point_cloud_size, dataset_folder=args.data_path)
    dataset_val = Objaverse(split='val', sdf_sampling=True, sdf_size=1024, surface_sampling=True, surface_size=args.point_cloud_size, dataset_folder=args.data_path)

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

    # if args.eval:
    #     test_stats = evaluate(data_loader_val, model, device)
    #     print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
    #     exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_iou = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        # test_stats = evaluate(data_loader_val, model, device)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        # if epoch % 5 == 0 or epoch + 1 == args.epochs:
        #     # test_stats = evaluate(data_loader_val, model, device)

        #     # print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
        #     # max_iou = max(max_iou, test_stats["iou"])
        #     # print(f'Max iou: {max_iou:.2f}%')

        #     # if log_writer is not None:
        #     #     # log_writer.add_scalar('perf/test_iou', test_stats['iou'], epoch)
        #     #     log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                     # **{f'test_{k}': v for k, v in test_stats.items()},
        #                     'epoch': epoch,
        #                     'n_parameters': n_parameters}
        # else:
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