#!/bin/bash

# Change to the source directory
cd VecSetX/vecset

# Set PYTHONPATH to include the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Training command
torchrun \
    --nproc_per_node=1 \
    main_ae.py \
    --accum_iter=1 \
    --model learnable_vec1024x16_dim1024_depth24_nb \
    --output_dir ../../output/ae/phase1_test \
    --log_dir ../../output/ae/phase1_test \
    --num_workers 4 \
    --point_cloud_size 8192 \
    --batch_size 2 \
    --epochs 10 \
    --warmup_epochs 1 --blr 5e-5 --clip_grad 1 \
    --device cuda
