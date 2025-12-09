#!/bin/bash
#SBATCH --job-name=test_data_gen
#SBATCH --account=project_2016526
#SBATCH --partition=gpumedium
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/test_data_gen_%j.log
#SBATCH --error=logs/test_data_gen_%j.err

# Test script for new trimesh-based data generation
# Tests on a SINGLE file to verify headless compatibility

echo "========================================"
echo "Testing Headless Data Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================"

# Setup environment
module load python-data/3.12-25.09
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

# Display Python info
python --version
python -c "import trimesh; print(f'trimesh version: {trimesh.__version__}')"

# Test directory
TEST_OUTPUT="/scratch/project_2016517/junjie/dataset/test_trimesh_npz"
mkdir -p $TEST_OUTPUT

# Clean any old test data
rm -f $TEST_OUTPUT/*.npz

echo ""
echo "Starting test data generation (1 file)..."
echo "----------------------------------------"

cd /projappl/project_2016517/JunjieCheng/VecSetX

# Run on SINGLE file for quick verification
python prepare_data.py \
   --input_dir /scratch/project_2016517/junjie/dataset/repaired_shape \
   --output_dir $TEST_OUTPUT \
   --num_surface_points 50000 \
   --num_vol_points 50000 \
   --classes 10 \
   --batch_size 10000 \
   --n_workers 32 \
   --file_workers 1 \
   --start_idx 0 \
   --end_idx 1

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Test Results"
echo "========================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Data generation SUCCESSFUL (no display errors!)"
    
    echo ""
    echo "Generated files:"
    ls -lh $TEST_OUTPUT/*.npz
    
    echo ""
    echo "Running quality check..."
    python check_data_quality.py $TEST_OUTPUT
    
else
    echo "❌ Data generation FAILED with exit code $EXIT_CODE"
    echo "Check error log above"
fi

echo ""
echo "Test completed at $(date)"
