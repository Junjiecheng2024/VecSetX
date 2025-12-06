#!/usr/bin/env python
"""
Debug script to verify if the label consistency fix is working
"""
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python debug_labels.py <path_to_npz>")
    sys.exit(1)

npz_path = sys.argv[1]
data = np.load(npz_path)

vol_sdf_all = data.get('vol_sdf_all', None)  # This won't exist in saved data
vol_sdf = data['vol_sdf'].flatten()
vol_labels = data['vol_labels'].flatten()

print(f"=== Debugging {npz_path} ===")
print(f"Total vol points: {len(vol_sdf)}")
print(f"Vol SDF range: [{vol_sdf.min():.4f}, {vol_sdf.max():.4f}]")
print(f"Vol labels unique: {np.unique(vol_labels)}")
print()

# Check 1: Points with SDF < 0 (should be inside)
inside_mask = vol_sdf < 0
n_inside = inside_mask.sum()
print(f"[Inside points] SDF < 0: {n_inside} ({n_inside/len(vol_sdf)*100:.1f}%)")

# Among inside points, how many have label == 0?
inside_with_no_label = np.logical_and(inside_mask, vol_labels == 0).sum()
print(f"  - Inside but No Label (label=0): {inside_with_no_label} ({inside_with_no_label/n_inside*100:.1f}%)")
print(f"  - Inside with Label (label>0): {n_inside - inside_with_no_label} ({(n_inside-inside_with_no_label)/n_inside*100:.1f}%)")
print()

# Check 2: Points with SDF >= 0 (should be outside)
outside_mask = vol_sdf >= 0
n_outside = outside_mask.sum()
print(f"[Outside points] SDF >= 0: {n_outside} ({n_outside/len(vol_sdf)*100:.1f}%)")

# Among outside points, how many have label > 0?
outside_with_label = np.logical_and(outside_mask, vol_labels > 0).sum()
print(f"  - Outside but Has Label (label>0): {outside_with_label} ({outside_with_label/n_outside*100:.1f}%)")
print(f"  - Outside with No Label (label=0): {n_outside - outside_with_label} ({(n_outside-outside_with_label)/n_outside*100:.1f}%)")
print()

# The FIX should ensure:
# 1. Most inside points have labels (inside_with_no_label should be LOW)
# 2. ALL outside points have no labels (outside_with_label should be 0)

if inside_with_no_label / n_inside > 0.1:
    print(f"❌ POTENTIAL ISSUE: {inside_with_no_label/n_inside*100:.1f}% of inside points have no label!")
    print("   This suggests the label assignment fix may not be working.")
else:
    print(f"✅ Good: Only {inside_with_no_label/n_inside*100:.1f}% of inside points lack labels")

if outside_with_label / n_outside > 0.01:
    print(f"❌ LABEL LEAK: {outside_with_label/n_outside*100:.1f}% of outside points have labels!")
    print("   This violates the geometry-label consistency principle.")
else:
    print(f"✅ Good: Only {outside_with_label/n_outside*100:.1f}% of outside points have labels")
