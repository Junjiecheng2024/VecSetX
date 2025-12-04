#!/bin/bash
#SBATCH --job-name=prepare_data_rtree
#SBATCH --account=project_2016517
#SBATCH --partition=small
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/prepare_data_rtree_%j.out
#SBATCH --error=logs/prepare_data_rtree_%j.err

# Create logs directory
mkdir -p logs

# Activate conda environment
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

echo "========================================="
echo "Data Preparation with rtree (Perfect SDF)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Memory: 64GB"
echo "========================================="

# Verify rtree installation
python -c "import rtree; print('rtree version:', rtree.__version__)"

# Set paths
INPUT_DIR=/scratch/project_2016517/junjie/dataset/repaired_shape
OUTPUT_DIR=/scratch/project_2016517/junjie/dataset/repaired_npz_perfect
SCRIPT_PATH=/projappl/project_2016517/JunjieCheng/VecSetX/prepare_data_batch.py

# Create output directory
mkdir -p $OUTPUT_DIR

# Process all files with rtree-based SDF computation
python $SCRIPT_PATH \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --num_surface_points 50000 \
    --num_vol_points 50000 \
    --classes 10 \
    --start_idx 0 \
    --end_idx 998

echo "========================================="
echo "Data Generation Completed!"
echo "========================================="

# Count generated files
NUM_FILES=$(ls $OUTPUT_DIR/*.npz 2>/dev/null | wc -l)
echo "Generated $NUM_FILES / 998 files"

# Quality check on 3 random samples
echo ""
echo "========================================="
echo "Data Quality Check (Random Samples)"
echo "========================================="

python -c "
import numpy as np
import glob
import os
import random

npz_files = sorted(glob.glob('$OUTPUT_DIR/*.npz'))
if len(npz_files) >= 3:
    samples = random.sample(npz_files, 3)
    
    for npz_file in samples:
        data = np.load(npz_file)
        vol_sdf = data['vol_sdf']
        near_sdf = data['near_sdf']
        
        print(f'\n{os.path.basename(npz_file)}:')
        print(f'  vol_sdf  - range: ({vol_sdf.min():.3f}, {vol_sdf.max():.3f}), mean: {vol_sdf.mean():.3f}, pos: {(vol_sdf>0).sum()/vol_sdf.size*100:.1f}%')
        print(f'  near_sdf - range: ({near_sdf.min():.3f}, {near_sdf.max():.3f}), mean: {near_sdf.mean():.3f}, pos: {(near_sdf>0).sum()/near_sdf.size*100:.1f}%')
        
        # Check if data looks good
        vol_pos_ratio = (vol_sdf > 0).sum() / vol_sdf.size
        near_pos_ratio = (near_sdf > 0).sum() / near_sdf.size
        
        if 0.2 < vol_pos_ratio < 0.8 and 0.2 < near_pos_ratio < 0.8:
            print(f'  ✅ Quality: GOOD')
        else:
            print(f'  ⚠️ Quality: Check needed')
else:
    print('Not enough files generated yet')
"

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
