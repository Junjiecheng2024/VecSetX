#!/bin/bash
#SBATCH -A project_2016517
#SBATCH --job-name=dryad_gen
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dryad_gen_%j.out
#SBATCH --error=logs/dryad_gen_%j.err

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
# Dryad has only 22 files, so we don't need massive parallelism.
# --file_workers 10: Process 10 files in parallel
echo "Starting Dryad data generation job on $(hostname)"
python prepare_data.py \
    --input_dir /scratch/project_2016517/junjie/dataset/dryad_nii \
    --output_dir /scratch/project_2016517/junjie/dataset/dryad_npz \
    --pattern "label_*.nii.gz" \
    --classes 16 \
    --file_workers 10 \
    --n_workers 0

echo "Job finished"
