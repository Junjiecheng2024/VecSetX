#!/bin/bash
#SBATCH --job-name=gen_perfect_data
#SBATCH --account=project_2016517
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/gen_perfect_data_%j.out
#SBATCH --error=logs/gen_perfect_data_%j.err

# Create logs directory
mkdir -p logs

# Activate environment
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

echo "========================================================================"
echo "Perfect Data Generation with rtree"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32GB"
echo "Start time: $(date)"
echo "========================================================================"

# Verify environment
echo ""
echo "Environment Check:"
python -c "
import trimesh
import rtree
print(f'✓ Trimesh: {trimesh.__version__}')
print(f'✓ rtree: {rtree.__version__}')

# Quick test
import numpy as np
mesh = trimesh.creation.box()
test = mesh.contains([[0, 0, 0], [10, 0, 0]])
print(f'✓ trimesh.contains() works: {test}')
"

echo ""
echo "========================================================================"
echo "Starting Data Generation (998 files)"
echo "========================================================================"

# Set paths
INPUT_DIR=/scratch/project_2016517/junjie/dataset/repaired_shape
OUTPUT_DIR=/scratch/project_2016517/junjie/dataset/repaired_npz
SCRIPT=/projappl/project_2016517/JunjieCheng/VecSetX/prepare_data_perfect.py

# Process all files in batches
for start in 0 100 200 300 400 500 600 700 800 900; do
    end=$((start + 100))
    
    echo ""
    echo "--------------------------------------------------------------------"
    echo "Processing batch: files $start to $((end-1))"
    echo "Time: $(date)"
    echo "--------------------------------------------------------------------"
    
    python $SCRIPT \
        --input_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR \
        --num_surface_points 50000 \
        --num_vol_points 50000 \
        --classes 10 \
        --batch_size 5000 \
        --start_idx $start \
        --end_idx $end
    
    # Progress update
    NUM_FILES=$(ls $OUTPUT_DIR/*.npz 2>/dev/null | wc -l)
    echo "Batch $start-$((end-1)) completed"
    echo "Total files so far: $NUM_FILES / 998"
done

echo ""
echo "========================================================================"
echo "Data Generation Completed!"
echo "========================================================================"
echo "End time: $(date)"

# Final statistics
FINAL_COUNT=$(ls $OUTPUT_DIR/*.npz 2>/dev/null | wc -l)
echo ""
echo "Final file count: $FINAL_COUNT / 998"

# Quality check on random samples
echo ""
echo "========================================================================"
echo "Quality Check (Random Samples)"
echo "========================================================================"

python -c "
import numpy as np
import glob
import random

npz_files = sorted(glob.glob('$OUTPUT_DIR/*.npz'))
print(f'Total files: {len(npz_files)}\n')

if len(npz_files) >= 5:
    samples = random.sample(npz_files, min(5, len(npz_files)))
    
    all_good = True
    for npz_file in samples:
        data = np.load(npz_file)
        vol_sdf = data['vol_sdf']
        near_sdf = data['near_sdf']
        
        vol_pos = (vol_sdf > 0).sum() / vol_sdf.size
        near_pos = (near_sdf > 0).sum() / near_sdf.size
        near_neg = (near_sdf < 0).sum() / near_sdf.size
        
        print(f'{npz_file.split(\"/\")[-1]}:')
        print(f'  vol_sdf:  pos={vol_pos*100:.1f}%, mean={vol_sdf.mean():.3f}')
        print(f'  near_sdf: pos={near_pos*100:.1f}%, neg={near_neg*100:.1f}%, mean={near_sdf.mean():.3f}')
        
        if near_pos > 0.2 and near_neg > 0.2:
            print(f'  ✅ Quality: PERFECT')
        else:
            print(f'  ⚠️ Quality: Check needed')
            all_good = False
        print()
    
    if all_good:
        print('='*70)
        print('✅✅✅ ALL SAMPLES PERFECT! Data quality is excellent!')
        print('='*70)
    else:
        print('⚠️ Some samples may have issues')
else:
    print('Not enough files to check')
"

echo ""
echo "========================================================================"
echo "Job completed successfully!"
echo "========================================================================"
