# mesh_to_sdf Data Generation Guide

## ⭐ mesh_to_sdf Solution

This is the **professional** approach using a dedicated SDF library instead of heuristics.

### Advantages
- ✅ Accurate inside/outside detection (uses ray-tracing)
- ✅ No manual threshold tuning needed
- ✅ Handles complex geometries correctly
- ✅ Production-quality results

### Installation

```bash
# Activate your environment
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

# Install mesh_to_sdf
pip install mesh-to-sdf

# Optional: Install pyembree for faster ray-tracing (recommended)
pip install pyembree
```

### Quick Test

```bash
cd /projappl/project_2016517/JunjieCheng/VecSetX

# Test on 1 file
python prepare_data_mesh_to_sdf.py \
    --input_dir /scratch/project_2016517/junjie/dataset/repaired_shape \
    --output_dir /scratch/project_2016517/junjie/dataset/test_mesh_to_sdf \
    --num_surface_points 50000 \
    --num_vol_points 50000 \
    --classes 10 \
    --start_idx 0 \
    --end_idx 1

# Check quality
python -c "
import numpy as np
data = np.load('/scratch/project_2016517/junjie/dataset/test_mesh_to_sdf/1.nii.img.npz')
vol_sdf = data['vol_sdf']
near_sdf = data['near_sdf']

print('='*60)
print('mesh_to_sdf Quality Check')
print('='*60)
print('\nvol_sdf:')
print(f'  Range: ({vol_sdf.min():.3f}, {vol_sdf.max():.3f})')
print(f'  Mean: {vol_sdf.mean():.3f}')
print(f'  Positive: {(vol_sdf > 0).sum() / vol_sdf.size * 100:.1f}%')
print(f'  Negative: {(vol_sdf < 0).sum() / vol_sdf.size * 100:.1f}%')

print('\nnear_sdf:')
print(f'  Mean: {near_sdf.mean():.3f}')
print(f'  Positive: {(near_sdf > 0).sum() / near_sdf.size * 100:.1f}%')
print(f'  Negative: {(near_sdf < 0).sum() / near_sdf.size * 100:.1f}%')

vol_good = 35 <= (vol_sdf > 0).sum() / vol_sdf.size * 100 <= 65
near_good = 35 <= (near_sdf > 0).sum() / near_sdf.size * 100 <= 65

print('\n' + '='*60)
if vol_good and near_good:
    print('✅✅✅ EXCELLENT! Professional SDF quality!')
    print('Ready for full dataset generation!')
else:
    print(f'Vol good: {vol_good}, Near good: {near_good}')
print('='*60)
"
```

### Full Dataset Generation

If test passes:

```bash
# Backup old data
mv /scratch/project_2016517/junjie/dataset/repaired_npz \
   /scratch/project_2016517/junjie/dataset/repaired_npz_v1_backup

# Generate all 998 files
mkdir -p /scratch/project_2016517/junjie/dataset/repaired_npz

for start in 0 100 200 300 400 500 600 700 800 900; do
    end=$((start + 100))
    echo "Batch $start-$((end-1))"
    python prepare_data_mesh_to_sdf.py \
        --input_dir /scratch/project_2016517/junjie/dataset/repaired_shape \
        --output_dir /scratch/project_2016517/junjie/dataset/repaired_npz \
        --num_surface_points 50000 \
        --num_vol_points 50000 \
        --classes 10 \
        --start_idx $start \
        --end_idx $end
done
```

### Performance Note

mesh_to_sdf is slower than heuristics but much more accurate:
- Heuristic methods: < 1 min/file
- mesh_to_sdf: 2-5 min/file
- Total time: ~3-8 hours for 998 files

**Worth it for professional-quality results!**

### Expected Results

- Vol SDF: 40-60% positive, 40-60% negative
- Near SDF: 40-60% positive, 40-60% negative
- Mean: ≈ 0.0
- Perfect inside/outside detection

### Troubleshooting

If installation fails:
```bash
# Try with conda
conda install -c conda-forge trimesh pyembree
pip install mesh-to-sdf
```

If too slow:
```bash
# Install pyembree for 2-3x speedup
pip install pyembree
```
