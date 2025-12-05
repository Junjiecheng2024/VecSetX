import os
import glob
import argparse
import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
from scipy.spatial import cKDTree
import gc

def get_args():
    parser = argparse.ArgumentParser(description="Convert .nii.gz to .npz (Lite v3 - Aggressive)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_surface_points", type=int, default=50000)
    parser.add_argument("--num_vol_points", type=int, default=50000)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=5000)
    return parser.parse_args()

def compute_sdf_aggressive(query_points, mesh_vertices, mesh_faces, batch_size=5000):
    """
    V3: Aggressive SDF computation targeting 50/50 distribution
    关键改进：threshold=0.5，更激进地判断内部
    """
    n_points = len(query_points)
    all_sdf = np.zeros(n_points, dtype=np.float32)
    
    # Build KDTree
    tree = cKDTree(mesh_vertices)
    
    # Mesh statistics
    mesh_center = mesh_vertices.mean(axis=0)
    mesh_std = mesh_vertices.std(axis=0).mean()
    
    # Get mesh bounding box for better estimation
    mesh_min = mesh_vertices.min(axis=0)
    mesh_max = mesh_vertices.max(axis=0)
    mesh_size = mesh_max - mesh_min
    
    # Process in batches
    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        batch_points = query_points[i:end]
        
        # Find nearest surface distance
        dists, idxs = tree.query(batch_points, k=3)  # k=3 for more context
        avg_dist = dists.mean(axis=1)  # Average of 3 nearest
        
        # Multi-criteria decision
        
        # Criterion 1: Very aggressive threshold (0.5 instead of 0.8)
        to_center = batch_points - mesh_center
        dist_to_center = np.linalg.norm(to_center, axis=1)
        threshold = 0.5  # 非常激进！
        inside_aggressive = dist_to_center < avg_dist * threshold
        
        # Criterion 2: Distance relative to mesh spread
        inside_by_spread = dist_to_center < mesh_std * 2.0
        
        # Criterion 3: Within bounding box consideration
        # 如果点在 bounding box 内部的 60% 区域，更可能是内部
        normalized_pos = (batch_points - mesh_min) / mesh_size
        in_inner_box = np.all((normalized_pos > 0.2) & (normalized_pos < 0.8), axis=1)
        
        # Combined: 满足任意两个条件就算内部
        votes = inside_aggressive.astype(int) + inside_by_spread.astype(int) + in_inner_box.astype(int)
        is_inside = votes >= 2  # Majority voting
        
        # Compute signed distance
        signs = np.where(is_inside, -1.0, 1.0)
        all_sdf[i:end] = avg_dist.astype(np.float32) * signs
        
    return all_sdf

def process_file(file_path, args):
    try:
        # Load NIfTI
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Extract meshes
        surface_points_list = []
        surface_labels_list = []
        total_surface_area = 0
        meshes = {}
        
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
                print(f"    Warning: Class {c} failed: {e}")
                continue

        if total_surface_area == 0:
            print(f"    Error: No surface found")
            return

        # Sample surface points
        for c, mesh in meshes.items():
            n_samples = max(1, int(args.num_surface_points * (mesh.area / total_surface_area)))
            points, _ = trimesh.sample.sample_surface(mesh, n_samples)
            surface_points_list.append(points)
            
            label = np.zeros((n_samples, args.classes), dtype=np.float32)
            label[:, c-1] = 1.0
            surface_labels_list.append(label)
            
        surface_points = np.concatenate(surface_points_list, axis=0)
        surface_labels = np.concatenate(surface_labels_list, axis=0)
        
        # Normalize to [-1, 1]
        shifts = (surface_points.max(axis=0) + surface_points.min(axis=0)) / 2
        surface_points = surface_points - shifts
        max_dist = np.max(np.linalg.norm(surface_points, axis=1))
        
        if max_dist > 0:
            scale_factor = 1 / max_dist
            surface_points *= scale_factor
        else:
            scale_factor = 1.0
        
        # Combine all meshes
        combined_verts = []
        combined_faces = []
        vertex_offset = 0
        
        for c, mesh in meshes.items():
            v = (mesh.vertices - shifts) * scale_factor
            combined_verts.append(v)
            combined_faces.append(mesh.faces + vertex_offset)
            vertex_offset += len(v)
            
        all_verts = np.concatenate(combined_verts)
        all_faces = np.concatenate(combined_faces)
        
        # Generate query points
        vol_points = np.random.uniform(-1, 1, (args.num_vol_points, 3)).astype(np.float32)
        
        # Compute SDF with V3 aggressive algorithm
        print(f"    Computing vol_sdf ({len(vol_points)} points)...")
        vol_sdf = compute_sdf_aggressive(vol_points, all_verts, all_faces, batch_size=args.batch_size)
        
        # Near surface points
        near_points = surface_points + np.random.normal(0, 0.01, surface_points.shape).astype(np.float32)
        near_points = np.clip(near_points, -1, 1)
        
        print(f"    Computing near_sdf ({len(near_points)} points)...")
        near_sdf = compute_sdf_aggressive(near_points, all_verts, all_faces, batch_size=args.batch_size)
        
        # Quality check
        vol_pos_ratio = (vol_sdf > 0).sum() / vol_sdf.size
        vol_neg_ratio = (vol_sdf < 0).sum() / vol_sdf.size
        near_pos_ratio = (near_sdf > 0).sum() / near_sdf.size
        near_neg_ratio = (near_sdf < 0).sum() / near_sdf.size
        
        print(f"    Quality: vol[+{vol_pos_ratio*100:.1f}% -{vol_neg_ratio*100:.1f}%] near[+{near_pos_ratio*100:.1f}% -{near_neg_ratio*100:.1f}%]")
        
        # Reshape
        vol_sdf = vol_sdf.reshape(-1, 1)
        near_sdf = near_sdf.reshape(-1, 1)
        
        # Save
        filename = os.path.basename(file_path).replace('.nii.gz', '.npz')
        save_path = os.path.join(args.output_dir, filename)
        
        np.savez(save_path, 
                 surface_points=surface_points.astype(np.float32), 
                 surface_labels=surface_labels.astype(np.float32),
                 vol_points=vol_points.astype(np.float32), 
                 vol_sdf=vol_sdf.astype(np.float32),
                 near_points=near_points.astype(np.float32),
                 near_sdf=near_sdf.astype(np.float32))
        
        print(f"    ✓ Saved {filename}")
        
        del meshes, surface_points, vol_points, near_points
        del vol_sdf, near_sdf, all_verts, all_faces
        gc.collect()

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.nii.gz")))
    print(f"Found {len(files)} files.")
    
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(files)
    files = files[start:end]
    
    print(f"Processing {len(files)} files (index {start} to {end-1})")
    print(f"Settings: surface={args.num_surface_points}, vol={args.num_vol_points}, batch={args.batch_size}")
    print(f"V3 Aggressive: threshold=0.5, k=3 neighbors, majority voting")
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
