import os
import glob
import argparse
import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
import gc

try:
    import mesh_to_sdf
    MESH_TO_SDF_AVAILABLE = True
except ImportError:
    MESH_TO_SDF_AVAILABLE = False

def get_args():
    parser = argparse.ArgumentParser(description="Multi-Class Data Generation (Consistent SDF + Labels)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_surface_points", type=int, default=50000)
    parser.add_argument("--num_vol_points", type=int, default=50000)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=5000)
    return parser.parse_args()

def compute_sdf_mesh_to_sdf(query_points, mesh):
    """
    Computes Signed Distance using mesh_to_sdf (High Quality).
    SDF < 0: Inside
    SDF > 0: Outside
    """
    if not MESH_TO_SDF_AVAILABLE:
        raise ImportError("mesh_to_sdf required")
    
    # mesh_to_sdf can be slow for large queries.
    # But it's robust.
    sdf_values = mesh_to_sdf.mesh_to_sdf(
        mesh, 
        query_points,
        surface_point_method='sample',
        sign_method='normal', # Reliable for marching cubes meshes
        scan_count=100,
        scan_resolution=400
    )
    return sdf_values.astype(np.float32)

def process_file(file_path, args):
    try:
        if not MESH_TO_SDF_AVAILABLE:
            print(f"    ✗ Skipped: mesh_to_sdf not available")
            return
            
        # Load NIfTI
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Extract meshes
        meshes = {}
        total_surface_area = 0
        
        # Temporary lists for surface sampling
        surface_points_list = []
        surface_labels_list = []
        
        for c in range(1, args.classes + 1):
            mask = (data == c)
            if np.sum(mask) == 0:
                continue
                
            try:
                verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                meshes[c] = mesh
                total_surface_area += mesh.area
            except Exception as e:
                print(f"    Warning: Class {c} failed extract: {e}")
                continue

        if total_surface_area == 0:
            print(f"    Error: No surface found")
            return

        # 1. Surface Sampling (Proportional)
        for c, mesh in meshes.items():
            ratio = mesh.area / total_surface_area
            n_surf = max(1, int(args.num_surface_points * ratio))
            
            points, _ = trimesh.sample.sample_surface(mesh, n_surf)
            surface_points_list.append(points)
            
            label = np.zeros((n_surf, args.classes), dtype=np.float32)
            label[:, c-1] = 1.0
            surface_labels_list.append(label)
            
        surface_points = np.concatenate(surface_points_list, axis=0)
        surface_labels = np.concatenate(surface_labels_list, axis=0)
        
        # Normalize
        shifts = (surface_points.max(axis=0) + surface_points.min(axis=0)) / 2
        surface_points = surface_points - shifts
        max_dist = np.max(np.linalg.norm(surface_points, axis=1))
        scale_factor = 1 / max_dist if max_dist > 0 else 1.0
        surface_points *= scale_factor
        
        # Scale Meshes for SDF calculation
        scaled_meshes = {}
        for c, mesh in meshes.items():
            v = (mesh.vertices - shifts) * scale_factor
            scaled_meshes[c] = trimesh.Trimesh(vertices=v, faces=mesh.faces)
            
        # 2. Generate Query Points
        vol_points = np.random.uniform(-1, 1, (args.num_vol_points, 3)).astype(np.float32)
        
        # Near Points: Jittered surface
        near_points = surface_points + np.random.normal(0, 0.01, surface_points.shape).astype(np.float32)
        near_points = np.clip(near_points, -1, 1)
        
        # 3. Compute Multi-Class SDFs (The Robust Way)
        # We calculate SDF for EACH class.
        # This gives us (N, C) matrix of SDF values.
        # Union SDF = min(SDF_1, SDF_2, ...)
        # Label = argmin(SDF_k) (if min < 0)
        
        print(f"    Computing Per-Class SDFs...")
        
        # --- Volume Points ---
        n_vol = len(vol_points)
        vol_sdf_all = np.full((n_vol, args.classes), 99.0, dtype=np.float32) # Initialize with large dist
        
        for c in scaled_meshes.keys():
            # print(f"      Class {c}...")
            # Compute SDF for this class
            sdf_c = compute_sdf_mesh_to_sdf(vol_points, scaled_meshes[c])
            vol_sdf_all[:, c-1] = sdf_c 
            
        # Combine
        # min along class axis
        vol_sdf = np.min(vol_sdf_all, axis=1) # (N,)
        
        # Labels
        # If min_sdf < 0, then label is argmin+1. Else 0.
        vol_labels = np.zeros(n_vol, dtype=np.int8)
        min_indices = np.argmin(vol_sdf_all, axis=1) # indices 0..(C-1)
        
        is_inside = vol_sdf < 0
        vol_labels[is_inside] = min_indices[is_inside] + 1
        
        # --- Near Points ---
        # Same logic
        n_near = len(near_points)
        near_sdf_all = np.full((n_near, args.classes), 99.0, dtype=np.float32)
        
        for c in scaled_meshes.keys():
            sdf_c = compute_sdf_mesh_to_sdf(near_points, scaled_meshes[c])
            near_sdf_all[:, c-1] = sdf_c
            
        near_sdf = np.min(near_sdf_all, axis=1)
        near_labels = np.zeros(n_near, dtype=np.int8)
        min_indices_near = np.argmin(near_sdf_all, axis=1)
        is_inside_near = near_sdf < 0
        near_labels[is_inside_near] = min_indices_near[is_inside_near] + 1
        
        # Reshape SDF
        vol_sdf = vol_sdf.reshape(-1, 1)
        near_sdf = near_sdf.reshape(-1, 1)
        
        # Quality Check (Self-Consistency is guaranteed by construction logic, but let's log)
        # Consistency: (SDF < 0) == (Label > 0)
        # By definition: is_inside = vol_sdf < 0; labels[is_inside] = ... + 1. So Label > 0.
        # Unless vol_sdf == 0? Unlikely with float.
        # Consistency is mathematically 100% here.
        
        pos_ratio = (vol_sdf > 0).mean()
        print(f"    SDF Balance: {pos_ratio*100:.1f}% Positive (Outside)")
        
        # Save
        filename = os.path.basename(file_path).replace('.nii.gz', '.npz')
        save_path = os.path.join(args.output_dir, filename)
        
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
        
        print(f"    ✓ Saved {filename}")
        
        del meshes, scaled_meshes
        del vol_points, vol_sdf, vol_labels, vol_sdf_all
        del near_points, near_sdf, near_labels, near_sdf_all
        gc.collect()

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    args = get_args()
    
    if not MESH_TO_SDF_AVAILABLE:
        print("\nERROR: mesh_to_sdf required!")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.nii.gz")))
    print(f"Found {len(files)} files.")
    
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(files)
    files = files[start:end]
    
    print(f"Processing {len(files)} files (index {start} to {end-1})")
    print(f"Algorithm: Per-Class mesh_to_sdf -> Min Pooling (Guaranteed Consistency)")
    print("="*70)
    
    for i, f in enumerate(files):
        print(f"\n[{start+i+1}/{end}] {os.path.basename(f)}")
        process_file(f, args)
        
        if (i + 1) % 10 == 0:
            gc.collect()
            print(f"  → {i+1}/{len(files)} completed")
    
    print(f"\n{'='*70}")
    print(f"✅ Completed {len(files)} files!")

if __name__ == "__main__":
    main()
