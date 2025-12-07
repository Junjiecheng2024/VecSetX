#!/usr/bin/env python
"""
Debug consistency issue in generated data
"""
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python debug_consistency.py <path_to_npz>")
    sys.exit(1)

npz_path = sys.argv[1]
data = np.load(npz_path)

vol_sdf = data['vol_sdf'].flatten()
vol_labels = data['vol_labels'].flatten()

print(f"=== Debugging {npz_path} ===\n")

# Basic stats
print(f"Total vol points: {len(vol_sdf)}")
print(f"Vol SDF > 0: {(vol_sdf > 0).sum()} ({(vol_sdf > 0).mean()*100:.1f}%)")
print(f"Vol labels > 0: {(vol_labels > 0).sum()} ({(vol_labels > 0).mean()*100:.1f}%)")

# Consistency check
inside_but_bg = np.logical_and(vol_sdf < 0, vol_labels == 0)
outside_but_cls = np.logical_and(vol_sdf >= 0, vol_labels > 0)

print(f"\n=== Consistency Issues ===")
print(f"Inside (SDF<0) but BG (label=0): {inside_but_bg.sum()} ({inside_but_bg.mean()*100:.1f}%)")
print(f"Outside (SDF>=0) but Class (label>0): {outside_but_cls.sum()} ({outside_but_cls.mean()*100:.1f}%)")
print(f"Consistency: {(1.0 - (inside_but_bg.mean() + outside_but_cls.mean()))*100:.1f}%")

# Detailed breakdown
print(f"\n=== Detailed Breakdown ===")
inside_points = (vol_sdf < 0).sum()
outside_points = (vol_sdf >= 0).sum()

# Among inside points
if inside_points > 0:
    inside_with_label = np.logical_and(vol_sdf < 0, vol_labels > 0).sum()
    inside_without_label = np.logical_and(vol_sdf < 0, vol_labels == 0).sum()
    print(f"\nInside points (SDF < 0): {inside_points}")
    print(f"  - With label (>0): {inside_with_label} ({inside_with_label/inside_points*100:.1f}%)")
    print(f"  - Without label (=0): {inside_without_label} ({inside_without_label/inside_points*100:.1f}%)")

# Among outside points
if outside_points > 0:
    outside_with_bg = np.logical_and(vol_sdf >= 0, vol_labels == 0).sum()
    outside_with_label = np.logical_and(vol_sdf >= 0, vol_labels > 0).sum()
    print(f"\nOutside points (SDF >= 0): {outside_points}")
    print(f"  - BG label (=0): {outside_with_bg} ({outside_with_bg/outside_points*100:.1f}%)")
    print(f"  - Class label (>0): {outside_with_label} ({outside_with_label/outside_points*100:.1f}%)")

# Sample some problematic cases
print(f"\n=== Sample Problematic Cases ===")
problematic = np.where(outside_but_cls)[0][:5]
print(f"\nOutside but has class label (first 5):")
for idx in problematic:
    print(f"  Point {idx}: SDF={vol_sdf[idx]:.4f}, Label={vol_labels[idx]}")

problematic2 = np.where(inside_but_bg)[0][:5]
print(f"\nInside but no label (first 5):")
for idx in problematic2:
    print(f"  Point {idx}: SDF={vol_sdf[idx]:.4f}, Label={vol_labels[idx]}")
