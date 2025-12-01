#!/bin/bash
#SBATCH --job-name=vecset_phase1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load necessary modules (if any)
# module load cuda/11.8  # Example, adjust based on cluster

# Activate environment
# source /path/to/venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=8

# WandB API Key (Optional: set it here or in your environment)
# export WANDB_API_KEY=your_key_here

# Create logs directory
mkdir -p logs

# Run training with torchrun
torchrun --nproc_per_node=4 --master_port=29500 VecSetX/vecset/main_ae.py \
    --batch_size 64 \
    --model learnable_vec1024x16_dim1024_depth24_nb \
    --point_cloud_size 8192 \
    --input_dim 13 \
    --epochs 800 \
    --data_path /home/user/persistent/vecset/data_npz \
    --output_dir output/ae/phase1_production \
    --log_dir output/ae/phase1_production \
    --wandb
