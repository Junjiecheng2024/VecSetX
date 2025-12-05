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
    parser = argparse.ArgumentParser(description="Smart SDF computation with adaptive methods")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_surface_points", type=int, default=50000)
    parser.add_argument("--num_vol_points", type=int, default=50000)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2000)
    return parser.parse_args()

def compute_sdf_smart(query_points, mesh, batch_size=2000):
    """
    Smart SDF: Use rtree only for ambiguous points, KDTree for obvious cases
    """
    n_points = len(query_points)
    all_sdf = np.zeros(n_points, dtype=np.float32)
    
    # Build KDTree for fast nearest neighbor
    tree = cKDTree(mesh.vertices)
    
    # Get mesh bounds for quick inside/outside estimation
    mesh_min = mesh.vertices.min(axis=0)
    mesh_max = mesh.vertices.max(axis=0)
    mesh_center = (mesh_min + mesh_max) / 2
    mesh_radius = np.linalg.norm(mesh_max - mesh_center)
    
    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        batch = query_points[i:end]
        
        # Step 1: Get distances
        dists, idxs = tree.query(batch, k=3)
        avg_dist = dists.mean(axis=1)
        
        # Step 2: Quick classification
        dist_to_center = np.linalg.norm(batch - mesh_center, axis=1)
        
        # Obviously outside (far from mesh)
        clearly_outside = dist_to_center > mesh_radius * 1.2
        
        # Ambiguous (need accurate check)
        ambiguous = ~clearly_outside
        
        # Step 3: Use rtree only for ambiguous points
        signs = np.ones(len(batch), dtype=np.float32)
        
        if ambiguous.sum() > 0:
            try:
                # Only check ambiguous points with rtree
                is_inside = mesh.contains(batch[ambiguous])
                signs[ambiguous] = np.where(is_inside, -1.0, 1.0)
            except:
                # Fallback: use distance heuristic
                signs[ambiguous] = np.where(
                    dist_to_center[ambiguous] < mesh_radius * 0.9,
                    -1.0, 1.0
                )
        
        all_sdf[i:end] = avg_dist * signs
    
    return all_sdf

def process_file(file_path, args):
    try:
        print(f"  Loading NIfTI...")
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Extract meshes
        surface_points_list = []
        surface_labels_list = []
        total_surface_area = 0
        meshes = {}
        
        print(f"  Extracting meshes...")
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
                continue
        
        if total_surface_area == 0:
            print(f"    Error: No surface found")
            return
        
        print(f"  Found {len(meshes)} classes, sampling surface...")
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
        
        # Normalize
        shifts = (surface_points.max(axis=0) + surface_points.min(axis=0)) / 2
        surface_points = surface_points - shifts
        max_dist = np.max(np.linalg.norm(surface_points, axis=1))
        
        if max_dist > 0:
            scale_factor = 1 / max_dist
            surface_points *= scale_factor
        else:
            scale_factor = 1.0
        
        # Combine meshes
        print(f"  Combining meshes...")
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
        combined_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
        
        print(f"  Mesh stats: {len(all_verts)} verts, {len(all_faces)} faces")
        
        # Generate query points
        vol_points = np.random.uniform(-1, 1, (args.num_vol_points, 3)).astype(np.float32)
        
        # Smart SDF computation
        print(f"  Computing SDF for {len(vol_points)} vol points (smart mode)...")
        vol_sdf = compute_sdf_smart(vol_points, combined_mesh, batch_size=args.batch_size)
        
        near_points = surface_points + np.random.normal(0, 0.01, surface_points.shape).astype(np.float32)
        near_points = np.clip(near_points, -1, 1)
        
        print(f"  Computing SDF for {len(near_points)} near points (smart mode)...")
        near_sdf = compute_sdf_smart(near_points, combined_mesh, batch_size=args.batch_size)
        
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
        
        print(f"  ✓ Saved {filename}")
        
        del combined_mesh, meshes, surface_points, vol_points, near_points
        del vol_sdf, near_sdf, all_verts, all_faces
        gc.collect()
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
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
    print("="*70)
    
    import time
    for i, f in enumerate(files):
        start_time = time.time()
        print(f"\n[{start+i+1}/{end}] {os.path.basename(f)}")
        process_file(f, args)
        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.1f}s")
        
        if (i + 1) % 5 == 0:
            gc.collect()
            print(f"\n  → {i+1}/{len(files)} completed")
    
    print(f"\n{'='*70}")
    print(f"✅ Completed {len(files)} files!")

if __name__ == "__main__":
    main()
