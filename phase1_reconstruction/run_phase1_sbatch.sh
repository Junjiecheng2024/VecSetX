#!/bin/bash
#SBATCH --job-name=vecset-p1
#SBATCH --account=project_2016526
#SBATCH --time=36:00:00
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:a100:4
#SBATCH -o /scratch/project_2016517/JunjieCheng/VecSetX/logs/phase1_%j.out
#SBATCH -e /scratch/project_2016517/JunjieCheng/VecSetX/logs/phase1_%j.err

# ============================================================
# VecSet Phase 1: Multi-Class Cardiac Reconstruction
# 
# 4x A100 GPU Training with torchrun
# All 10 cardiac classes: Myo, LA, LV, RA, RV, Aorta, PA, LAA, Coronary, PV
# ============================================================

set -e

echo "============================================================"
echo "🫀 VecSet Phase 1: Multi-Class Cardiac Reconstruction"
echo "============================================================"

# Work directories
WORKDIR="/scratch/project_2016517/JunjieCheng"
PROJECTDIR="/projappl/project_2016517/JunjieCheng/VecSetX"
DATA_DIR="/scratch/project_2016517/JunjieCheng/dataset/vecset_phase1"
VECSET_ROOT="/scratch/project_2016517/JunjieCheng/VecSetX"
OUTDIR="$VECSET_ROOT/outputs/phase1"
IMG="/scratch/project_2016517/JunjieCheng/pytorch.sif"

# Environment variables
export PYTHONUSERBASE=$WORKDIR/pyuser
export PIP_CACHE_DIR=$WORKDIR/pip-cache
export TMPDIR="/scratch/project_2016526/JunjieCheng/tmp"
export XDG_CACHE_HOME=$WORKDIR/.cache
export TORCH_HOME=$WORKDIR/.cache/torch
export HF_HOME=$WORKDIR/.cache/huggingface
export MPLCONFIGDIR=$WORKDIR/.config/matplotlib
export OMP_NUM_THREADS=10
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=$(hostname)

# WandB configuration
export WANDB_API_KEY="d6891a1bb4397a24519ef1b36091aa1b77ea67e1"
export WANDB_DIR=$VECSET_ROOT/wandb
export WANDB_CONFIG_DIR=$VECSET_ROOT/.config/wandb

# Create directories
mkdir -p $PYTHONUSERBASE $PIP_CACHE_DIR $TMPDIR
mkdir -p $XDG_CACHE_HOME $TORCH_HOME $HF_HOME $MPLCONFIGDIR
mkdir -p $WANDB_DIR $WANDB_CONFIG_DIR
mkdir -p "$OUTDIR"
mkdir -p "/scratch/project_2016517/JunjieCheng/VecSetX/logs"

cd "$PROJECTDIR"

# Training configuration
BATCH_SIZE=1
EPOCHS=800
ACCUM_ITER=16
LR=5e-4
WARMUP=10
SAVE_FREQ=10
NUM_WORKERS=8

# Model configuration
MODEL="learnable_vec1024x16_dim1024_depth24_nb"
PC_SIZE=8192
SDF_SIZE=4096

# All 10 cardiac classes
CLASSES="1 2 3 4 5 6 7 8 9 10"

# Randomly set MASTER_PORT based on job ID to avoid conflicts
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

echo "📊 Configuration:"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUTDIR"
echo "  Model:      $MODEL"
echo "  Classes:    $CLASSES (all 10 cardiac classes)"
echo "  Batch size: $BATCH_SIZE x 4 GPUs x $ACCUM_ITER accum = $((BATCH_SIZE * 4 * ACCUM_ITER)) effective"
echo "  Epochs:     $EPOCHS"
echo "  LR:         $LR"
echo "  Container:  $IMG"
echo "  Master:     $MASTER_ADDR:$MASTER_PORT"
echo "============================================================"

# Run training with torchrun (4 GPUs)
srun apptainer exec --nv \
  -B /scratch:/scratch \
  -B /projappl:/projappl \
  --env PYTHONUSERBASE=$PYTHONUSERBASE \
  --env PYTHONPATH=$PROJECTDIR \
  --env TORCH_HOME=$TORCH_HOME \
  --env HF_HOME=$HF_HOME \
  --env MPLCONFIGDIR=$MPLCONFIGDIR \
  --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
  --env TMPDIR=$TMPDIR \
  --env MASTER_ADDR=$MASTER_ADDR \
  --env MASTER_PORT=$MASTER_PORT \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  --env WANDB_DIR=$WANDB_DIR \
  --env WANDB_CONFIG_DIR=$WANDB_CONFIG_DIR \
  "$IMG" \
  torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $PROJECTDIR/phase1_reconstruction/train.py \
      --data_path $DATA_DIR \
      --output_dir $OUTDIR \
      --log_dir $OUTDIR/logs \
      --model $MODEL \
      --point_cloud_size $PC_SIZE \
      --sdf_size $SDF_SIZE \
      --batch_size $BATCH_SIZE \
      --epochs $EPOCHS \
      --accum_iter $ACCUM_ITER \
      --blr $LR \
      --warmup_epochs $WARMUP \
      --save_freq $SAVE_FREQ \
      --num_workers $NUM_WORKERS \
      --classes $CLASSES \
      --pin_mem \
      --use_wandb \
      --wandb_project vecset-phase1

echo "============================================================"
echo "✅ Phase 1 training completed!"
echo "Output: $OUTDIR"
echo "============================================================"
