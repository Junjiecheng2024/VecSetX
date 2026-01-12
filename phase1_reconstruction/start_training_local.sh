#!/bin/bash
# Phase 1 Local Training Script (for testing)
# Usage: ./start_training_local.sh

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="outputs/phase1/logs/training_${TIMESTAMP}.log"

mkdir -p outputs/phase1/logs

echo "Phase 1: Multi-Class Cardiac Reconstruction"
echo "Log file: $LOG_FILE"

nohup python3 -u phase1_reconstruction/train.py \
  --data_path /path/to/vecset_phase1 \
  --output_dir outputs/phase1 \
  --log_dir outputs/phase1/logs \
  --model learnable_vec1024x16_dim1024_depth24_nb \
  --point_cloud_size 8192 \
  --sdf_size 4096 \
  --batch_size 2 \
  --epochs 800 \
  --accum_iter 8 \
  --blr 5e-4 \
  --warmup_epochs 10 \
  --save_freq 10 \
  --num_workers 8 \
  --classes 1 2 3 4 5 6 7 8 9 10 \
  --pin_mem \
  --use_wandb \
  --wandb_project vecset-phase1 \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started!"
echo "Log file: $LOG_FILE"
echo "PID: $PID"
echo ""
echo "To monitor training:"
echo "  tail -f $LOG_FILE"
