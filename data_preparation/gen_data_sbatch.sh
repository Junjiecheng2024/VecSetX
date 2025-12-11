#!/bin/bash
#SBATCH -A project_2016517
#SBATCH --job-name=vecset_gen_data
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gen_data_%j.out
#SBATCH --error=logs/gen_data_%j.err

# Load environment
module load python-data/3.12-25.09
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

# Setup paths
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=1

# Working Directory (Project Root)
cd /projappl/project_2016517/JunjieCheng/VecSetX/data_preparation

# Create logs directory
mkdir -p logs

# Run generation
# --file_workers 20: Process 20 files in parallel (matching cpus-per-task)
# --n_workers 0: Inner parallelization disabled when file_workers > 1
echo "Starting data generation job on $(hostname)"
python prepare_data.py \
    --input_dir /scratch/project_2016517/junjie/dataset/repaired_shape \
    --output_dir /scratch/project_2016517/junjie/dataset/repaired_npz \
    --file_workers 20 \
    --n_workers 0

echo "Job finished"
