
import os
import glob
import argparse
import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
from scipy import ndimage
import gc

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Multi-Class Voxel-based Scalable Generation (Balanced)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_surface_points", type=int, default=50000)
    parser.add_argument("--num_vol_points", type=int, default=50000)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100000) 
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--n_workers", type=int, default=0) 
    parser.add_argument("--file_workers", type=int, default=1)
    
    # Sampling Config
    parser.add_argument("--min_surface_per_class", type=int, default=2000, help="Minimum surface points per class")
    parser.add_argument("--vol_per_class", type=int, default=2500, help="Volume points per class (Inside)")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Voxel-based SDF & Interpolation
# -----------------------------------------------------------------------------

def compute_grid_sdf(mask):
    """
    Compute signed distance field on the voxel grid using Euclidean Distance Transform.
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
    
    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1
    
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
    
    values = c0 * (1 - zd) + c1 * zd
    return values

# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------

def process_file(file_path, args):
    try:
        filename = os.path.basename(file_path).replace('.nii.gz', '.npz')
        save_path = os.path.join(args.output_dir, filename)
        
        # 1. Load Data
        img = nib.load(file_path)
        data = img.get_fdata(dtype=np.float32) # Shape: (X, Y, Z)
        
        # 2. Precompute SDF Grids and Extract Meshes
        sdf_grids = {}
        meshes = {}
        total_surface_area = 0
        failed_classes = []
        
        # Create a combined mask for "Inside" volume sampling
        # Ideally, union of all classes
        combined_mask = np.zeros(data.shape, dtype=bool)
        
        for c in range(1, args.classes + 1):
            mask = (data == c)
            if np.sum(mask) == 0:
                failed_classes.append(c)
                continue
            
            combined_mask |= mask
            
            # Mesh
            try:
                verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
                # Note: verts are (x, y, z) matching data indices
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                meshes[c] = mesh
                total_surface_area += mesh.area
            except Exception as e:
                print(f"    ⚠️ Class {c}: Marching cubes failed - {e}")
                failed_classes.append(c)
                continue
            
            # SDF Grid
            sdf_grids[c] = compute_grid_sdf(mask)

        if total_surface_area == 0:
            print(f"    Error: No surface found in {filename}")
            return None

        # 3. Sample Surface Points & Normalize Info
        surface_points_list = []
        surface_labels_list = []
        surface_normals_list = [] # Store normals for Near Points
        
        for c, mesh in meshes.items():
            # Hybrid Sampling: Quota + Weighted
            # 1. Quota
            quota = args.min_surface_per_class
            
            # 2. Weighted (from remaining pool)
            remaining_pool = max(0, args.num_surface_points - (len(meshes) * quota))
            weighted_share = int(remaining_pool * (mesh.area / total_surface_area))
            
            n_samples = quota + weighted_share
            
            # sample_surface returns (points, face_index)
            p, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
            
            # Get normals from faces
            # mesh.face_normals is (N_faces, 3)
            # We want normals for each sampled point. Face normal is good approx.
            n = mesh.face_normals[face_idx]
            
            surface_points_list.append(p)
            surface_normals_list.append(n)
            
            l = np.zeros((n_samples, args.classes), dtype=np.float32)
            l[:, c-1] = 1.0
            surface_labels_list.append(l)
            
        surface_points_vox = np.concatenate(surface_points_list, axis=0)
        surface_labels = np.concatenate(surface_labels_list, axis=0)
        surface_normals = np.concatenate(surface_normals_list, axis=0) # Normalized unit vectors
        
        # --- Normalization Parameters ---
        shifts = (surface_points_vox.max(axis=0) + surface_points_vox.min(axis=0)) / 2
        centered_points = surface_points_vox - shifts
        max_dist = np.max(np.linalg.norm(centered_points, axis=1))
        scale_factor = 1.0 / max_dist if max_dist > 0 else 1.0
        
        # Normalize Surface Points
        surface_points = centered_points * scale_factor
        
        # 4. Generate Volume Points (Balanced Stratified)
        # Goal: ~25k Inside (Stratified by Class), ~25k Random(Outside bias)
        
        n_vol_total = args.num_vol_points
        n_in_target = n_vol_total // 2
        # Target per class for inside points
        n_in_per_class = args.vol_per_class 
        # Note: If n_in_per_class * classes != n_in_target, it might vary slightly.
        # But roughly 2500 * 10 = 25000.
        
        vol_points_in_list = []
        
        # A. Inside Points (Stratified per Class)
        for c in range(1, args.classes + 1):
            # Get mask for specific class
            mask_c = (label_img == c)
            inside_indices = np.argwhere(mask_c)
            
            if len(inside_indices) > 0:
                # Sample 2500 points for this class
                idxs = np.random.choice(len(inside_indices), n_in_per_class, replace=True)
                chosen_indices = inside_indices[idxs].astype(np.float32)
                
                # Jitter
                jitter = np.random.uniform(-0.5, 0.5, chosen_indices.shape).astype(np.float32)
                p_vox = chosen_indices + jitter
                p_norm = (p_vox - shifts) * scale_factor
                vol_points_in_list.append(p_norm)
            else:
                 # If class is missing in this patient, skip (or distribute to others? simplified: skip)
                 pass
                 
        if len(vol_points_in_list) > 0:
            vol_points_in = np.vstack(vol_points_in_list)
        else:
            vol_points_in = np.zeros((0, 3), dtype=np.float32)

        # B. Outside/Random Points (Fill the rest)
        n_current = len(vol_points_in)
        n_out = max(0, n_vol_total - n_current)
        
        # Uniform sampling in [-1, 1]
        vol_points_out = np.random.uniform(-1, 1, (n_out, 3)).astype(np.float32)
        
        vol_points = np.vstack([vol_points_in, vol_points_out])
        
        # 5. Generate Near Points (Balanced 50/50)
        # Strategy: Use surface normals to push strictly In/Out
        # n_near = num_surface_points (from list size matching)
        
        # We reuse the sampled surface points to generate near points
        # Half push out, half push in
        N_surf = len(surface_points)
        
        # Random noise magnitude: e.g., Gaussian(0, 0.01)
        # Using abs() to ensure direction control
        eps = np.abs(np.random.normal(0, 0.01, (N_surf, 1))).astype(np.float32)
        
        # 50% indices
        half = N_surf // 2
        
        # First half: Push Out (Normal is pointing out) -> P + eps * N
        # Second half: Push In (Normal pointing out) -> P - eps * N
        # This assumes normals point OUT. Marching Cubes usually does.
        
        near_points = np.zeros_like(surface_points)
        
        # Assign
        # Note: surface_normals are unit vectors.
        # surface_normals calculated from original mesh (unscaled). 
        # But direction is same in scaled space (uniform scaling).
        
        # Points 0..half: Out
        near_points[:half] = surface_points[:half] + eps[:half] * surface_normals[:half]
        
        # Points half..end: In
        near_points[half:] = surface_points[half:] - eps[half:] * surface_normals[half:]
        
        # Clip to ensure valid range
        near_points = np.clip(near_points, -1, 1)
        
        # 6. Compute SDFs using Trilinear Interpolation
        def get_sdf_and_labels(query_points_norm):
            points_vox = query_points_norm / scale_factor + shifts
            N = len(points_vox)
            sdf_all_classes = np.full((N, args.classes), 99.0, dtype=np.float32)
            
            for c, grid in sdf_grids.items():
                sdf_all_classes[:, c-1] = trilinear_interpolation(grid, points_vox)
            
            sdf_all_classes *= scale_factor
            union_sdf = np.min(sdf_all_classes, axis=1)
            
            labels = np.zeros(N, dtype=np.int8)
            is_inside = union_sdf < 0
            indices = np.where(is_inside)[0]
            
            if len(indices) > 0:
                inside_sdfs = sdf_all_classes[indices] 
                deepest_class_idx = np.argmin(inside_sdfs, axis=1) 
                labels[indices] = deepest_class_idx + 1 
                
            return union_sdf.reshape(-1, 1), labels

        vol_sdf, vol_labels = get_sdf_and_labels(vol_points)
        near_sdf, near_labels = get_sdf_and_labels(near_points)
        
        # 7. Save
        vp = (vol_sdf > 0).mean()
        np_pos = (near_sdf > 0).mean()
        
        del sdf_grids, meshes, data, img, combined_mask
        gc.collect()
        
        print(f"    Vol+: {vp*100:.1f}% | Near+: {np_pos*100:.1f}% | Saved {filename}")
        
        np.savez(save_path, 
                 surface_points=surface_points.astype(np.float32), 
                 surface_labels=surface_labels.astype(np.float32),
                 vol_points=vol_points.astype(np.float32), 
                 vol_sdf=vol_sdf.astype(np.float32),
                 vol_labels=vol_labels,
                 near_points=near_points.astype(np.float32),
                 near_sdf=near_sdf.astype(np.float32),
                 near_labels=near_labels
                 )
        return filename

    except Exception as e:
        print(f"    ✗ Failed {os.path.basename(file_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    args = get_args()
    print("Using Voxel-based EDT SDF (Super Fast & Balanced)")
    
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.nii.gz")))
    
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(files)
    files = files[start:end]
    
    print(f"Processing {len(files)} files.")
    
    if args.file_workers > 1:
        from multiprocessing import Pool
        from functools import partial
        print(f"Parallel file workers: {args.file_workers}")
        func = partial(process_file, args=args)
        with Pool(args.file_workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(func, files)):
                if (i+1) % 50 == 0:
                    print(f"Progress: {i+1}/{len(files)}")
                    gc.collect()
    else:
        for i, f in enumerate(files):
            print(f"[{i+1}/{len(files)}] {os.path.basename(f)}")
            process_file(f, args)
            if (i+1) % 10 == 0:
                gc.collect()
    
    print("Done.")

if __name__ == "__main__":
    main()
