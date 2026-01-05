#!/bin/bash
#SBATCH -A project_2016526
#SBATCH --job-name=ecg_datagen_s3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p medium
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_2016517/JunjieCheng/PhysioNet-Digitization_of_ECG_Images/logs/datagen_s3_%j.out
#SBATCH --error=/scratch/project_2016517/JunjieCheng/PhysioNet-Digitization_of_ECG_Images/logs/datagen_s3_%j.err

set -euo pipefail

# ================================================================================
# 🔥 Stage 3 Data Generation - Kaggle 信号 + 竞赛匹配退化
# ================================================================================
#
# 使用 Kaggle train 的真实临床 ECG 信号
# 应用与竞赛数据完全匹配的退化效果:
#   - 0003/0004: 彩色/黑白扫描
#   - 0005/0006: 手机/屏幕拍照
#   - 0009/0010: 污渍/严重损坏
#   - 0011/0012: 发霉彩色/黑白
#   - 额外: 皱褶、折痕、阴影、旋转、手写字
#
# ================================================================================

WORKDIR=/scratch/project_2016517/JunjieCheng
PROJECTDIR=/projappl/project_2016517/JunjieCheng/PhysioNet-Digitization_of_ECG_Images
OUTDIR=/scratch/project_2016526/JunjieCheng/dataset/synthetic_v3_kaggle
KAGGLE_DIR=/scratch/project_2016526/JunjieCheng/dataset/physionet.org/files/ecg-arrhythmia
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
MAX_SAMPLES=25000   # Stage 3 目标样本数
NUM_WORKERS=32      # 并行工作进程数
SEED=2025123       # 随机种子

# 创建必要目录
mkdir -p "$OUTDIR" "$OUTDIR/images" "$OUTDIR/masks" "$OUTDIR/annotations"
mkdir -p "/scratch/project_2016517/JunjieCheng/PhysioNet-Digitization_of_ECG_Images/logs"
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

cd "$PROJECTDIR"

# ================================================================================
# 数据生成 v3.0 (Kaggle 信号 + 竞赛匹配退化)
# ================================================================================
echo "============================================================"
echo "🔥 Stage 3 Data Generation (Kaggle Signals)"
echo "============================================================"
echo "Kaggle dir: $KAGGLE_DIR"
echo "Output: $OUTDIR"
echo "Container: $IMG"
echo ""
echo "📊 Configuration:"
echo "  Max samples: $MAX_SAMPLES"
echo "  Workers: $NUM_WORKERS"
echo "  Seed: $SEED"
echo ""

apptainer exec \
  -B /scratch:/scratch \
  -B /projappl:/projappl \
  --env PYTHONUSERBASE=$PYTHONUSERBASE \
  --env PIP_CACHE_DIR=$PIP_CACHE_DIR \
  --env TMPDIR=$TMPDIR \
  --env XDG_CACHE_HOME=$XDG_CACHE_HOME \
  --env MPLCONFIGDIR=$MPLCONFIGDIR \
  --env HOME=$HOME \
  --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
  "$IMG" \
  bash -lc "
    set -e
    export PYTHONPATH=\"\$PYTHONPATH:$PROJECTDIR\"
    export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"
    
    echo 'Python: '\$(which python)
    echo 'Checking dependencies...'
    python -c 'import pandas, cv2, matplotlib; print(\"✅ All deps OK\")'
    
    python -u ECG/scripts/generate_data_v3.py \\
      --input_dir $KAGGLE_DIR \\
      --output_dir $OUTDIR \\
      --max_samples $MAX_SAMPLES \\
      --num_workers $NUM_WORKERS \\
      --seed $SEED
  "

echo ""
echo "============================================================"
echo "✅ Stage 3 data generation completed!"
echo "Output: $OUTDIR"
echo "Samples: $MAX_SAMPLES"
echo "============================================================"
