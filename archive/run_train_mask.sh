#!/bin/bash
#SBATCH --job-name=ecg-mask-model
#SBATCH --account=project_2016526
#SBATCH --time=36:00:00
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:4
#SBATCH -o /scratch/project_2016517/JunjieCheng/PhysioNet-Digitization_of_ECG_Images/logs/%x_%j.out
#SBATCH -e /scratch/project_2016517/JunjieCheng/PhysioNet-Digitization_of_ECG_Images/logs/%x_%j.err

# ============================================================
# Model A: Mask 专精模型训练
# 
# 专注于 Mask 预测精度
# 使用后处理管线提取信号
# ============================================================

set -e

echo "============================================================"
echo "🎯 Model A: Mask-Focused Training"
echo "============================================================"

# 工作目录
WORKDIR="/scratch/project_2016517/JunjieCheng"

# 🆕 关键环境变量 (让容器能找到 user-installed 包)
export PYTHONUSERBASE=$WORKDIR/pyuser
export PIP_CACHE_DIR=$WORKDIR/pip-cache
# 🆕 TMPDIR Redirect: Use project_2016526 for temp files (atomic save support)
export TMPDIR="/scratch/project_2016526/JunjieCheng/tmp"
export XDG_CACHE_HOME=$WORKDIR/.cache
export HOME=$WORKDIR

# 🆕 模型权重缓存目录 (避免写入只读 /users)
export TORCH_HOME=$WORKDIR/.cache/torch
export HF_HOME=$WORKDIR/.cache/huggingface

# WandB 设置 (避免写入只读 home)
export WANDB_API_KEY="d6891a1bb4397a24519ef1b36091aa1b77ea67e1"
export WANDB_DIR=$WORKDIR/wandb
export WANDB_CONFIG_DIR=$WORKDIR/.config/wandb
export MPLCONFIGDIR=$WORKDIR/.config/matplotlib

# 其他环境变量
export OMP_NUM_THREADS=10
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=$(hostname)


# 创建必要目录
mkdir -p $PYTHONUSERBASE $PIP_CACHE_DIR $TMPDIR
mkdir -p $XDG_CACHE_HOME $WANDB_DIR $WANDB_CONFIG_DIR $MPLCONFIGDIR
mkdir -p $TORCH_HOME $HF_HOME

# 路径
PROJECTDIR="/projappl/project_2016517/JunjieCheng/PhysioNet-Digitization_of_ECG_Images"
DATA_DIR="/scratch/project_2016526/JunjieCheng/dataset/synthetic_v3_kaggle"
OUTDIR="/scratch/project_2016526/JunjieCheng/outputs/mask_model_phase3"
IMG="/scratch/project_2016517/JunjieCheng/pytorch.sif"

# 创建日志目录
mkdir -p logs
mkdir -p "$OUTDIR"

# 🔥 Stage 2: 加载 Pretrain 权重
CKPT="/scratch/project_2016526/JunjieCheng/outputs/mask_model_phase3/20260104_170937/checkpoints/mask-best-epoch=39-val/dice=0.6113.ckpt"

echo "📊 Configuration:"
echo "  Image: $IMG"
echo "  Data: $DATA_DIR"
echo "  Output: $OUTDIR"
echo "============================================================"

cd "$PROJECTDIR"

# 检查数据目录
echo "Checking data directory..."
ls -la "$DATA_DIR" | head -5
echo "============================================================"

# Run training - Lightning handles multi-GPU via devices=4
export PYTHONPATH="$PYTHONPATH:$PROJECTDIR"

echo "GPUs visible: $(nvidia-smi -L 2>/dev/null | wc -l)"
# Randomly set MASTER_PORT based on job ID to avoid conflicts
export CONF_MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
echo "Using Master Port: $CONF_MASTER_PORT"

# 运行训练脚本
srun apptainer exec --nv \
  -B /scratch:/scratch \
  -B /projappl:/projappl \
  --env PYTHONUSERBASE=$PYTHONUSERBASE \
  --env PYTHONPATH=$PYTHONPATH \
  --env TORCH_HOME=$TORCH_HOME \
  --env HF_HOME=$HF_HOME \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  --env WANDB_DIR=$WANDB_DIR \
  --env MPLCONFIGDIR=$MPLCONFIGDIR \
  --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
  --env TMPDIR=$TMPDIR \
  "$IMG" \
  python -u ECG/training/train_mask_model.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTDIR \
    --devices 4 \
    --num_nodes 1 \
    --strategy ddp \
    --encoder efficientnet-b7 \
    --epochs 40 \
    --max_samples 20000 \
    --batch_size 3 \
    --accum_iter 48 \
    --lr 1e-5 \
    --image_height 1024 \
    --image_width 1408 \
    --num_workers 8 \
    --precision 16-mixed \
    --wandb_project ecg-mask-model \
    --ckpt $CKPT

echo "============================================================"
echo "✅ Model A training completed!"
echo "============================================================"
