#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=vecset_phase1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpumedium
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH -J vecset_phase1

# Load necessary modules (if any)
# module load cuda/11.8  # Example, adjust based on cluster

module load python-data/3.10-24.04
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=8


# Create logs directory
mkdir -p logs

# Run training with torchrun
cd /projappl/project_2016517/JunjieCheng/VecSetX

torchrun --nproc_per_node=4 --master_port=29500 VecSetX/vecset/main_ae.py \
    --batch_size 64 \
    --model learnable_vec1024x16_dim1024_depth24_nb \
    --point_cloud_size 8192 \
    --input_dim 13 \
    --epochs 800 \
    --data_path /scratch/project_2016517/junjie/dataset/repaired_npz \
    --output_dir output/ae/phase1_production \
    --log_dir output/ae/phase1_production \
    --wandb
