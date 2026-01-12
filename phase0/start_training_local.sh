#!/bin/bash
# Phase 0 Training Startup Script with WandB
# Usage: ./start_training_local.sh

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Ensure output directory exists
mkdir -p /home/user/persistent/VecSetX/outputs/phase0/logs

# Set WandB API Key and run training in background
WANDB_API_KEY="d6891a1bb4397a24519ef1b36091aa1b77ea67e1" \
nohup python3 -u phase0/train.py \
      --data_path /home/user/persistent/VecSetX/dataset/vecset_phase0_myo \
      --output_dir /home/user/persistent/VecSetX/outputs/phase0 \
      --log_dir /home/user/persistent/VecSetX/outputs/phase0/logs \
      --model learnable_vec1024x16_dim1024_depth24_nb \
      --point_cloud_size 8192 \
      --sdf_size 4096 \
      --batch_size 2 \
      --epochs 800 \
      --accum_iter 8 \
      --blr 5e-4 \
      --warmup_epochs 10 \
      --min_lr 1e-6 \
      --weight_decay 0.05 \
      --save_freq 10 \
      --num_workers 8 \
      --classes 1 \
      --pin_mem \
      --use_wandb \
      --wandb_project vecset-phase0 \
      > /home/user/persistent/VecSetX/outputs/phase0/logs/training_${TIMESTAMP}.log 2>&1 &

echo "Training started!"
echo "Log file: outputs/phase0/logs/training_${TIMESTAMP}.log"
echo "PID: $!"
echo ""
echo "To monitor training:"
echo "  tail -f outputs/phase0/logs/training_${TIMESTAMP}.log"
