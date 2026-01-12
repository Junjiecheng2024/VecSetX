#!/usr/bin/env python3
"""
Per-Class Binary Implicit Data Preparation (V2)

This script follows the advisor's recommendation:
- Each class is treated as an independent binary implicit task
- No one-hot encoding - class info is implicit from data grouping
- SDF/Eikonal loss is mathematically valid for each class
- Transformer will learn geometry, not classification

Output structure:
    output_dir/
        case_001/
            class_1/  (Myocardium)
                surface_pts.npy   (N, 3)
                near_pts.npy      (N, 3)
                near_sdf.npy      (N, 1)
                vol_pts.npy       (N, 3)
                vol_sdf.npy       (N, 1)
            class_2/  (LA)
                ...
            ...
        case_002/
            ...
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
from scipy import ndimage
import gc
import json


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CLASS_NAMES = {
    1: "Myocardium",
    2: "LA",
    3: "LV",
    4: "RA",
    5: "RV",
    6: "Aorta",
    7: "PA",
    8: "LAA",
    9: "Coronary",
    10: "PV",
    11: "SVC",
    12: "IVC",
    13: "CS",
    14: "RVW",
    15: "LAW",
    16: "PML",
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Per-Class Binary Implicit Data Preparation (V2)"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .nii.gz segmentation masks")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for per-class data")
    parser.add_argument("--num_surface_points", type=int, default=8192,
                        help="Surface points per class")
    parser.add_argument("--num_near_points", type=int, default=8192,
                        help="Near-surface points per class")
    parser.add_argument("--num_vol_points", type=int, default=8192,
                        help="Volume points per class")
    parser.add_argument("--classes", type=str, default="1,2,3,4,5,6,7,8,9,10",
                        help="Comma-separated list of class IDs to process (e.g., '2' or '2,3,4')")
    parser.add_argument("--near_std", type=float, default=0.01,
                        help="Std of Gaussian noise for near-surface points")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--file_workers", type=int, default=1)
    parser.add_argument("--pattern", type=str, default="*.nii.gz")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# SDF Computation (Voxel-based EDT)
# -----------------------------------------------------------------------------

def compute_grid_sdf(mask):
    """
    Compute signed distance field using Euclidean Distance Transform.
    Returns SDF where: negative = inside, positive = outside
    """
    dist_inside = ndimage.distance_transform_edt(mask)
    dist_outside = ndimage.distance_transform_edt(~mask)
    sdf = dist_outside - dist_inside
    return sdf.astype(np.float32)


def trilinear_interpolation(grid, points):
    """
    Interpolate 3D grid values at given points.
    grid: (D, H, W)
    points: (N, 3) in [0, D-1], order (z, y, x)
    """
    D, H, W = grid.shape
    
    z = points[:, 0]
    y = points[:, 1]
    x = points[:, 2]
    
    z0 = np.floor(z).astype(int)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    
    z0 = np.clip(z0, 0, D - 2)
    y0 = np.clip(y0, 0, H - 2)
    x0 = np.clip(x0, 0, W - 2)
    
    z1, y1, x1 = z0 + 1, y0 + 1, x0 + 1
    
    zd = z - z0
    yd = y - y0
    xd = x - x0
    
    c000 = grid[z0, y0, x0]
    c001 = grid[z0, y0, x1]
    c010 = grid[z0, y1, x0]
    c011 = grid[z0, y1, x1]
    c100 = grid[z1, y0, x0]
    c101 = grid[z1, y0, x1]
    c110 = grid[z1, y1, x0]
    c111 = grid[z1, y1, x1]
    
    c00 = c000 * (1 - xd) + c001 * xd
    c01 = c010 * (1 - xd) + c011 * xd
    c10 = c100 * (1 - xd) + c101 * xd
    c11 = c110 * (1 - xd) + c111 * xd
    
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd
    
    return c0 * (1 - zd) + c1 * zd


# -----------------------------------------------------------------------------
# Per-Class Processing
# -----------------------------------------------------------------------------

def process_single_class(mask, class_id, args, shifts, scale_factor, volume_shape):
    """
    Process a single class as a binary implicit task.
    
    Args:
        mask: Binary mask for this class (D, H, W)
        class_id: Class ID (1-indexed)
        args: Command line arguments
        shifts: Center shift for normalization
        scale_factor: Scale factor for normalization
        volume_shape: Original volume shape (D, H, W) for voxel-space sampling
    
    Returns:
        dict with surface_pts, near_pts, near_sdf, vol_pts, vol_sdf, stats
        or None if class has no valid surface
    """
    D, H, W = volume_shape
    
    # 1. Extract mesh using marching cubes
    try:
        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        if mesh.area < 1e-6:
            return None
    except Exception as e:
        print(f"      Marching cubes failed for class {class_id}: {e}")
        return None
    
    # 2. Compute SDF grid for this class
    sdf_grid = compute_grid_sdf(mask)
    
    # 3. Sample surface points
    n_surface = args.num_surface_points
    surface_pts_vox, face_idx = trimesh.sample.sample_surface(mesh, n_surface)
    
    # Normalize to [-1, 1]
    surface_pts = (surface_pts_vox - shifts) * scale_factor
    
    # 4. Generate near-surface points with SDF-based filtering for strict 50/50
    # Strategy: generate more points, then filter by SDF sign to ensure exact 50/50
    n_near = args.num_near_points
    n_half = n_near // 2
    
    # Oversample to have enough points after filtering
    oversample_factor = 3
    n_candidates = n_near * oversample_factor
    
    # Sample base surface points
    near_base_pts_vox, _ = trimesh.sample.sample_surface(mesh, n_candidates)
    near_base_pts = (near_base_pts_vox - shifts) * scale_factor
    
    # Random offset magnitude (Gaussian)
    eps = np.abs(np.random.normal(0, args.near_std, (n_candidates, 1))).astype(np.float32)
    
    # Random direction (unit vectors)
    random_dirs = np.random.randn(n_candidates, 3).astype(np.float32)
    random_dirs = random_dirs / (np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-8)
    
    # Generate candidate near points by random perturbation
    near_candidates = near_base_pts + eps * random_dirs
    near_candidates = np.clip(near_candidates, -1, 1)
    
    # Compute SDF for all candidates
    near_candidates_vox = near_candidates / scale_factor + shifts
    near_candidates_sdf = trilinear_interpolation(sdf_grid, near_candidates_vox)
    near_candidates_sdf = (near_candidates_sdf * scale_factor).reshape(-1, 1)
    
    # Split by SDF sign
    outside_mask = (near_candidates_sdf > 0).flatten()  # positive = outside
    inside_mask = ~outside_mask  # negative = inside
    
    outside_indices = np.where(outside_mask)[0]
    inside_indices = np.where(inside_mask)[0]
    
    # Sample exactly n_half from each group
    if len(outside_indices) >= n_half and len(inside_indices) >= n_half:
        # Ideal case: enough points in both groups
        chosen_outside = np.random.choice(outside_indices, n_half, replace=False)
        chosen_inside = np.random.choice(inside_indices, n_half, replace=False)
    else:
        # Fallback: use replacement if not enough
        chosen_outside = np.random.choice(outside_indices, n_half, replace=True) if len(outside_indices) > 0 else np.array([], dtype=int)
        chosen_inside = np.random.choice(inside_indices, n_half, replace=True) if len(inside_indices) > 0 else np.array([], dtype=int)
        
        # If one group is empty, fill from the other
        if len(chosen_outside) < n_half:
            extra_needed = n_half - len(chosen_outside)
            extra = np.random.choice(inside_indices, extra_needed, replace=True) if len(inside_indices) > 0 else np.array([], dtype=int)
            chosen_outside = np.concatenate([chosen_outside, extra]) if len(chosen_outside) > 0 else extra
        if len(chosen_inside) < n_half:
            extra_needed = n_half - len(chosen_inside)
            extra = np.random.choice(outside_indices, extra_needed, replace=True) if len(outside_indices) > 0 else np.array([], dtype=int)
            chosen_inside = np.concatenate([chosen_inside, extra]) if len(chosen_inside) > 0 else extra
    
    chosen_indices = np.concatenate([chosen_outside, chosen_inside])
    near_pts = near_candidates[chosen_indices]
    near_sdf = near_candidates_sdf[chosen_indices]
    
    # 5. Generate volume points (50% inside, 50% uniform in VOXEL space)
    n_vol = args.num_vol_points
    n_inside = n_vol // 2
    n_uniform = n_vol - n_inside
    
    # Inside points: sample from mask voxels
    inside_voxel_indices = np.argwhere(mask)
    if len(inside_voxel_indices) > 0:
        idxs = np.random.choice(len(inside_voxel_indices), n_inside, replace=True)
        inside_pts_vox = inside_voxel_indices[idxs].astype(np.float32)
        # Add jitter within voxel
        jitter = np.random.uniform(-0.5, 0.5, inside_pts_vox.shape).astype(np.float32)
        inside_pts_vox = inside_pts_vox + jitter
        inside_pts = (inside_pts_vox - shifts) * scale_factor
    else:
        inside_pts = np.zeros((0, 3), dtype=np.float32)
    
    # Uniform points: sample in VOXEL space, then normalize
    # This ensures points are within the actual volume bounds
    uniform_pts_vox = np.zeros((n_uniform, 3), dtype=np.float32)
    uniform_pts_vox[:, 0] = np.random.uniform(0, D, n_uniform)  # z
    uniform_pts_vox[:, 1] = np.random.uniform(0, H, n_uniform)  # y
    uniform_pts_vox[:, 2] = np.random.uniform(0, W, n_uniform)  # x
    uniform_pts = (uniform_pts_vox - shifts) * scale_factor
    
    # Combine volume points
    vol_pts = np.vstack([inside_pts, uniform_pts]) if len(inside_pts) > 0 else uniform_pts
    
    # Compute SDF for volume points
    vol_pts_vox = vol_pts / scale_factor + shifts
    vol_sdf = trilinear_interpolation(sdf_grid, vol_pts_vox)
    vol_sdf = (vol_sdf * scale_factor).reshape(-1, 1)
    
    # 6. Compute comprehensive statistics
    pos_ratio_near = (near_sdf > 0).mean() * 100
    neg_ratio_near = (near_sdf < 0).mean() * 100
    pos_ratio_vol = (vol_sdf > 0).mean() * 100
    
    stats = {
        'class_id': int(class_id),
        'class_name': CLASS_NAMES.get(class_id, f"Class{class_id}"),
        'n_surface': int(len(surface_pts)),
        'n_near': int(len(near_pts)),
        'n_vol': int(len(vol_pts)),
        'near_pos_ratio': float(pos_ratio_near),
        'near_neg_ratio': float(neg_ratio_near),
        'vol_pos_ratio': float(pos_ratio_vol),
        'near_sdf_min': float(near_sdf.min()),
        'near_sdf_max': float(near_sdf.max()),
        'near_sdf_mean': float(near_sdf.mean()),
        'vol_sdf_min': float(vol_sdf.min()),
        'vol_sdf_max': float(vol_sdf.max()),
        'vol_sdf_mean': float(vol_sdf.mean()),
        'mesh_area': float(mesh.area),
        'mesh_volume': float(mesh.volume) if mesh.is_volume else 0.0,
    }
    
    return {
        'surface_pts': surface_pts.astype(np.float32),
        'near_pts': near_pts.astype(np.float32),
        'near_sdf': near_sdf.astype(np.float32),
        'vol_pts': vol_pts.astype(np.float32),
        'vol_sdf': vol_sdf.astype(np.float32),
        'stats': stats
    }


# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------

def process_file(file_path, args):
    """Process a single .nii.gz file and save per-class data."""
    try:
        case_name = os.path.basename(file_path).replace('.nii.gz', '')
        case_dir = os.path.join(args.output_dir, case_name)
        
        # Load segmentation mask (use integer comparison to avoid float precision issues)
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.int16)  # Convert to int for reliable comparison
        
        # Compute global normalization parameters from ALL class surfaces combined
        all_surface_points = []
        
        # Parse class IDs from comma-separated string
        class_ids = [int(c.strip()) for c in args.classes.split(',')]
        
        for c in class_ids:
            mask = (data == c)  # Now safe: int == int comparison
            if np.sum(mask) == 0:
                continue
            try:
                verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
                all_surface_points.append(verts)
            except:
                continue
        
        if len(all_surface_points) == 0:
            print(f"    ✗ No valid classes in {case_name}")
            return None
        
        all_surface_points = np.vstack(all_surface_points)
        
        # Normalization: center and scale to [-1, 1] unit sphere
        shifts = (all_surface_points.max(axis=0) + all_surface_points.min(axis=0)) / 2
        centered = all_surface_points - shifts
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        scale_factor = 1.0 / max_dist if max_dist > 0 else 1.0
        
        del all_surface_points, centered
        
        # Get volume shape for voxel-space sampling
        volume_shape = data.shape
        
        # Process each class independently
        processed_classes = []
        all_class_stats = {}
        
        # Parse class IDs from comma-separated string
        class_ids = [int(c.strip()) for c in args.classes.split(',')]
        
        for c in class_ids:
            mask = (data == c)  # Safe int comparison
            if np.sum(mask) == 0:
                continue
            
            class_data = process_single_class(mask, c, args, shifts, scale_factor, volume_shape)
            if class_data is None:
                continue
            
            # Create class directory
            class_dir = os.path.join(case_dir, f"class_{c}")
            os.makedirs(class_dir, exist_ok=True)
            
            # Save per-class data as separate .npy files
            np.save(os.path.join(class_dir, 'surface_pts.npy'), class_data['surface_pts'])
            np.save(os.path.join(class_dir, 'near_pts.npy'), class_data['near_pts'])
            np.save(os.path.join(class_dir, 'near_sdf.npy'), class_data['near_sdf'])
            np.save(os.path.join(class_dir, 'vol_pts.npy'), class_data['vol_pts'])
            np.save(os.path.join(class_dir, 'vol_sdf.npy'), class_data['vol_sdf'])
            
            # Save per-class stats.json
            stats = class_data['stats']
            with open(os.path.join(class_dir, 'stats.json'), 'w') as f:
                json.dump(stats, f, indent=2)
            
            all_class_stats[c] = stats
            
            class_name = CLASS_NAMES.get(c, f"Class{c}")
            print(f"      Class {c:2d} ({class_name:12s}): "
                  f"Near+: {stats['near_pos_ratio']:.1f}% | "
                  f"Vol+: {stats['vol_pos_ratio']:.1f}%")
            
            processed_classes.append(c)
        
        # Save metadata
        metadata = {
            'case_name': case_name,
            'shifts': shifts.tolist(),
            'scale_factor': float(scale_factor),
            'volume_shape': list(volume_shape),
            'classes': processed_classes,
            'num_surface_points': args.num_surface_points,
            'num_near_points': args.num_near_points,
            'num_vol_points': args.num_vol_points,
            'class_stats': {str(k): v for k, v in all_class_stats.items()},
        }
        
        with open(os.path.join(case_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"    ✓ Saved {case_name} ({len(processed_classes)} classes)")
        
        del data, img
        gc.collect()
        
        return case_name
    
    except Exception as e:
        print(f"    ✗ Failed {os.path.basename(file_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    args = get_args()
    
    print("=" * 60)
    print("Per-Class Binary Implicit Data Preparation (V2)")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Classes: {args.classes}")
    print(f"Points per class: surface={args.num_surface_points}, "
          f"near={args.num_near_points}, vol={args.num_vol_points}")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(files)
    files = files[start:end]
    
    print(f"Processing {len(files)} files...")
    print()
    
    if args.file_workers > 1:
        from multiprocessing import Pool
        from functools import partial
        
        func = partial(process_file, args=args)
        with Pool(args.file_workers) as pool:
            results = list(pool.imap(func, files))
    else:
        results = []
        for i, f in enumerate(files):
            print(f"[{i+1}/{len(files)}] {os.path.basename(f)}")
            result = process_file(f, args)
            results.append(result)
            if (i + 1) % 10 == 0:
                gc.collect()
    
    # Summary
    success = sum(1 for r in results if r is not None)
    print()
    print("=" * 60)
    print(f"Done! Successfully processed {success}/{len(files)} cases.")
    print("=" * 60)


if __name__ == "__main__":
    main()
