#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=vecset_gendata_p0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p medium
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project_2016517/JunjieCheng/VecSetX/logs/gendata_p0_%j.out
#SBATCH --error=/scratch/project_2016517/JunjieCheng/VecSetX/logs/gendata_p0_%j.err

set -euo pipefail

# ================================================================================
# 🫀 Phase 0 Data Generation - Myocardium Only (Single Class Experiment)
# ================================================================================
#
# Generate per-class implicit data for VecSetX
# Only Class 1 (Myocardium) for initial baseline experiment
#
# Data format (per case):
#   case_xxx/
#     class_1/
#       surface_pts.npy   (100000, 3)
#       near_pts.npy      (100000, 3)
#       near_sdf.npy      (100000, 1)
#       vol_pts.npy       (100000, 3)
#       vol_sdf.npy       (100000, 1)
#       stats.json
#     metadata.json
#
# ================================================================================

WORKDIR=/scratch/project_2016517/JunjieCheng
PROJECTDIR=/projappl/project_2016517/JunjieCheng/VecSetX
INPUT_DIR=/scratch/project_2016517/JunjieCheng/dataset/repaired_Public-Cardiac-CT-Dataset
OUTPUT_DIR=/scratch/project_2016517/JunjieCheng/dataset/vecset_phase0_myo
IMG=$WORKDIR/pytorch.sif

# ================================================================================
# 环境变量配置
# ================================================================================
export PYTHONUSERBASE=$WORKDIR/pyuser
export PIP_CACHE_DIR=$WORKDIR/pip-cache
export TMPDIR=$WORKDIR/pip-tmp
export XDG_CACHE_HOME=$WORKDIR/.cache
export MPLCONFIGDIR=$WORKDIR/.config/matplotlib
export HOME=$WORKDIR
export PATH="$PYTHONUSERBASE/bin:$PATH"
export OMP_NUM_THREADS=32

# ================================================================================
# 🔥 配置参数
# ================================================================================
NUM_SURFACE=100000    # 表面点数量
NUM_NEAR=100000       # 近表面点数量
NUM_VOL=100000        # 体积点数量
CLASSES=1             # 只处理 Myocardium (class 1)
FILE_WORKERS=16       # 并行处理文件数
PATTERN="*_mask.nii.gz"  # 文件匹配模式

# 创建必要目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "/scratch/project_2016517/JunjieCheng/VecSetX/logs"
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

cd "$PROJECTDIR"

# ================================================================================
# Phase 0 数据生成 (Myocardium Only)
# ================================================================================
echo "============================================================"
echo "🫀 Phase 0 Data Generation (Myocardium Only)"
echo "============================================================"
echo "Input dir:  $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Container:  $IMG"
echo ""
echo "📊 Configuration:"
echo "  Surface points: $NUM_SURFACE"
echo "  Near points:    $NUM_NEAR"
echo "  Volume points:  $NUM_VOL"
echo "  Classes:        $CLASSES (Myocardium only)"
echo "  File workers:   $FILE_WORKERS"
echo "  Pattern:        $PATTERN"
echo ""

apptainer exec \
  -B /scratch:/scratch \
  -B /projappl:/projappl \
  --env PYTHONUSERBASE=$PYTHONUSERBASE \
  --env PIP_CACHE_DIR=$PIP_CACHE_DIR \
  --env TMPDIR=$TMPDIR \
  --env XDG_CACHE_HOME=$XDG_CACHE_HOME \
  --env MPLCONFIGDIR=$MPLCONFIGDIR \
  --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
  "$IMG" \
  bash -lc "
    set -e
    export PYTHONPATH=\$PYTHONPATH:$PROJECTDIR
    export PATH=$PYTHONUSERBASE/bin:\$PATH
    
    echo 'Python: '\$(which python)
    echo 'Checking dependencies...'
    python -c 'import numpy, nibabel, trimesh, scipy; print(\"All deps OK\")'
    
    python -u $PROJECTDIR/data_preparation/prepare_data_v2.py \
      --input_dir $INPUT_DIR \
      --output_dir $OUTPUT_DIR \
      --num_surface_points $NUM_SURFACE \
      --num_near_points $NUM_NEAR \
      --num_vol_points $NUM_VOL \
      --classes $CLASSES \
      --file_workers $FILE_WORKERS \
      --pattern '*_mask.nii.gz'
  "

echo ""
echo "============================================================"
echo "✅ Phase 0 data generation completed!"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
