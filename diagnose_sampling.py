#!/usr/bin/env python
"""
Diagnose why certain classes are missing from vol_labels
"""
import numpy as np
import nibabel as nib
import sys

if len(sys.argv) < 2:
    print("Usage: python diagnose_sampling.py <nii_file> <npz_file>")
    sys.exit(1)

nii_path = sys.argv[1]
npz_path = sys.argv[2]

# Load original data
img = nib.load(nii_path)
data = img.get_fdata()

# Load generated data
npz = np.load(npz_path)
vol_labels = npz['vol_labels'].flatten()

print(f"=== Original Data Analysis ===")
total_voxels = data.size
for c in range(0, 11):
    n_voxels = (data == c).sum()
    pct = n_voxels / total_voxels * 100
    print(f"Class {c}: {n_voxels:8d} voxels ({pct:5.2f}%)")

print(f"\n=== Vol Labels Sampling Results ===")
print(f"Total vol points: {len(vol_labels)}")
for c in range(0, 11):
    n_sampled = (vol_labels == c).sum()
    pct = n_sampled / len(vol_labels) * 100
    print(f"Class {c}: {n_sampled:5d} points ({pct:5.2f}%)")

print(f"\n=== Missing Classes ===")
original_classes = set([int(c) for c in np.unique(data)])
sampled_classes = set([int(c) for c in np.unique(vol_labels)])
missing = original_classes - sampled_classes
if missing:
    print(f"Classes present in original but missing in vol_labels: {sorted(missing)}")
    for c in sorted(missing):
        n_voxels = (data == c).sum()
        pct = n_voxels / total_voxels * 100
        print(f"  Class {c}: {n_voxels} voxels ({pct:.3f}% of total)")
else:
    print("✅ All classes successfully sampled!")
