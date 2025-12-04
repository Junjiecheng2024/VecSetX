#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --account=project_2016517
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/prepare_data_%j.out
#SBATCH --error=logs/prepare_data_%j.err

# Create logs directory
mkdir -p logs

# Activate conda environment
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

# Set paths
INPUT_DIR=/scratch/project_2016517/junjie/dataset/repaired_shape
OUTPUT_DIR=/scratch/project_2016517/junjie/dataset/repaired_npz
SCRIPT_PATH=/projappl/project_2016517/JunjieCheng/VecSetX/prepare_data_batch.py

echo "========================================="
echo "Data Preparation Job Started"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32GB"
echo "========================================="

# Process all files (0 to 998)
python $SCRIPT_PATH \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --num_surface_points 100000 \
    --num_vol_points 100000 \
    --classes 10 \
    --start_idx 0 \
    --end_idx 998

echo "========================================="
echo "Data Preparation Completed!"
echo "========================================="

# Verify results
NUM_FILES=$(ls $OUTPUT_DIR/*.npz 2>/dev/null | wc -l)
echo "Generated $NUM_FILES npz files"

# Quick data quality check on first file
python -c "
import numpy as np
import glob
import os

npz_files = sorted(glob.glob('$OUTPUT_DIR/*.npz'))
if npz_files:
    data = np.load(npz_files[0])
    vol_sdf = data['vol_sdf']
    print('\n=== Data Quality Check (First File) ===')
    print(f'File: {os.path.basename(npz_files[0])}')
    print(f'vol_sdf range: ({vol_sdf.min():.4f}, {vol_sdf.max():.4f})')
    print(f'vol_sdf mean: {vol_sdf.mean():.4f}')
    print(f'Positive ratio: {(vol_sdf > 0).sum() / vol_sdf.size * 100:.1f}%')
    print(f'Negative ratio: {(vol_sdf < 0).sum() / vol_sdf.size * 100:.1f}%')
    
    if abs(vol_sdf.mean()) < 0.5 and (vol_sdf > 0).sum() > 0:
        print('✅ Data looks good!')
    else:
        print('⚠️ Data may have issues')
"
