#!/bin/bash
#SBATCH --job-name=gen_phase1_data
#SBATCH --account=project_2016526
#SBATCH --partition=gpumedium
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=480G
#SBATCH --output=logs/gen_data_%j.log
#SBATCH --error=logs/gen_data_%j.err

# Full dataset generation using trimesh-based headless SDF computation
# Expected time: 3-8 hours for 998 files

echo "========================================"
echo "Phase 1 Full Dataset Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================"

# Setup
module load python-data/3.12-25.09
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

# Config
DATA_DIR="/scratch/project_2016517/junjie/dataset/repaired_shape"
OUTPUT_DIR="/scratch/project_2016517/junjie/dataset/repaired_npz"

# Create output dir
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo ""
echo "Configuration:"
echo "  Input:  $DATA_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  CPUs:   128"
echo "  Memory: 480G"
echo ""

cd /projappl/project_2016517/JunjieCheng/VecSetX

# Count total files
TOTAL_FILES=$(ls -1 $DATA_DIR/*.nii.gz | wc -l)
echo "Total files to process: $TOTAL_FILES"
echo ""

# Run data generation
# Using file_workers=16 for parallel file processing
# n_workers=8 for per-file parallelization (128/16 = 8)
python prepare_data.py \
   --input_dir $DATA_DIR \
   --output_dir $OUTPUT_DIR \
   --num_surface_points 50000 \
   --num_vol_points 50000 \
   --classes 10 \
   --batch_size 10000 \
   --n_workers 8 \
   --file_workers 16 \
   --start_idx 0 \
   --end_idx $TOTAL_FILES

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Generation Summary"
echo "========================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Full dataset generation SUCCESSFUL"
    
    GENERATED=$(ls -1 $OUTPUT_DIR/*.npz 2>/dev/null | wc -l)
    echo "Generated: $GENERATED / $TOTAL_FILES files"
    
    echo ""
    echo "Running quality check on full dataset..."
    python check_data_quality.py $OUTPUT_DIR
    
    echo ""
    echo "Updating CSV files..."
    python create_csv.py
    
else
    echo "❌ Generation FAILED with exit code $EXIT_CODE"
fi

echo ""
echo "Completed at $(date)"
