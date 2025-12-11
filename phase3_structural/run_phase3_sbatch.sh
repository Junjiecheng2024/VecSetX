#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=vecset_phase3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpumedium
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --output=logs/%j_phase3.out
#SBATCH --error=logs/%j_phase3.err
#SBATCH -J vecset_phase3

# Load necessary modules
module load python-data/3.12-25.09
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=8

# Create logs directory
mkdir -p logs

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Run training with OPTIMIZED hyperparameters
cd /projappl/project_2016517/JunjieCheng/VecSetX

torchrun --nproc_per_node=4 --master_port=29502 phase3_structural/train.py \
    --batch_size 2 \
    --accum_iter 8 \
    --model learnable_vec1024x16_dim1024_depth24_nb \
    --point_cloud_size 8192 \
    --input_dim 13 \
    --epochs 400 \
    --data_path /scratch/project_2016517/junjie/dataset/repaired_npz \
    --output_dir output/ae/phase3_structural \
    --log_dir output/ae/phase3_structural \
    --blr 2e-4 \
    --warmup_epochs 20 \
    --wandb \
    --partial_prob 0.8 \
    --min_remove 0 \
    --max_remove 0

