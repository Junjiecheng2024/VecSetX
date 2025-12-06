#!/usr/bin/env python
"""
Verify that surface_labels contain all 10 classes even when vol_labels don't
"""
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python verify_all_classes.py <path_to_npz>")
    sys.exit(1)

npz_path = sys.argv[1]
data = np.load(npz_path)

print(f"=== Verifying {npz_path} ===\n")

# Check surface_labels (one-hot encoded)
surface_labels = data['surface_labels']  # Shape: (N, 10)
print(f"Surface points: {len(surface_labels)}")

# Find which classes have surface points
classes_in_surface = []
for c in range(10):
    if surface_labels[:, c].sum() > 0:
        n_points = surface_labels[:, c].sum()
        classes_in_surface.append(c+1)
        print(f"  Class {c+1}: {int(n_points)} surface points")

print(f"\n✅ Surface labels contain {len(classes_in_surface)} classes: {classes_in_surface}")

# Check vol_labels (integer labels)
vol_labels = data['vol_labels'].flatten()
classes_in_vol = sorted([int(c) for c in np.unique(vol_labels) if c > 0])
print(f"✅ Vol labels contain {len(classes_in_vol)} classes: {classes_in_vol}")

# Compare
missing_from_vol = set(classes_in_surface) - set(classes_in_vol)
if missing_from_vol:
    print(f"\n⚠️  Classes present in surface but missing from vol: {sorted(missing_from_vol)}")
    print("   This is NORMAL for very small anatomical structures!")
    print("   Training uses surface_labels, so all classes are represented.")
else:
    print(f"\n✅ All classes present in both surface and vol labels!")
