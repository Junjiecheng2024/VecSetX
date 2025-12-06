#!/bin/bash
#SBATCH --job-name=data_gen
#SBATCH --account=project_2016517
#SBATCH --partition=medium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --output=logs/data_gen_%j.out
#SBATCH --error=logs/data_gen_%j.err

module purge
module load pytorch/2.4

# Create logs directory
mkdir -p logs

# Activate environment
source /projappl/project_2016517/junjie_conda/miniconda3/bin/activate junjieenv

cd /projappl/project_2016517/JunjieCheng/VecSetX

echo "=== Starting parallel data generation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Generate all 998 files with 8 parallel workers
# Each worker processes files sequentially, but 8 workers run simultaneously
python prepare_data_optimized.py \
    --input_dir /scratch/project_2016517/junjie/dataset/repaired_shape \
    --output_dir /scratch/project_2016517/junjie/dataset/repaired_npz \
    --vol_threshold 0.85 \
    --n_workers 32 \
    --file_workers 8

echo "=== Data generation complete ==="

# Final quality check
echo "=== Running quality check on sample files ==="
python check_data_quality.py /scratch/project_2016517/junjie/dataset/repaired_npz

echo "✅ All done!"
