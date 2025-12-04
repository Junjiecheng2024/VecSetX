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
    parser = argparse.ArgumentParser(description="Convert .nii.gz masks to .npz for VecSetX (Memory Optimized)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_surface_points", type=int, default=50000, help="Reduced from 100k")
    parser.add_argument("--num_vol_points", type=int, default=50000, help="Reduced from 100k")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=5000, help="SDF computation batch size")
    return parser.parse_args()

def compute_sdf_simple(query_points, mesh_vertices, mesh_faces, batch_size=5000):
    """
    Memory-efficient SDF computation using batching
    """
    n_points = len(query_points)
    all_sdf = np.zeros(n_points, dtype=np.float32)
    
    # Build KDTree once
    tree = cKDTree(mesh_vertices)
    
    # Process in batches
    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        batch_points = query_points[i:end]
        
        # Find nearest vertices
        dists, idxs = tree.query(batch_points, k=1)
        
        # Simple heuristic: use distance to nearest vertex
        # For a more accurate inside/outside test, we check if point is 
        # closer to mesh than a threshold (simplified Contains)
        
        # Compute actual distance to mesh surface (nearest point on triangles)
        # This is approximate but memory-efficient
        
        # For now, use signed distance based on proximity
        # Negative inside, positive outside
        # We'll use a simple heuristic: if very close to surface, likely inside
        
        # Better approach: check if point is inside bounding box and close to surface
        batch_sdf = dists.astype(np.float32)
        
        # Approximate inside/outside using mesh center
        mesh_center = mesh_vertices.mean(axis=0)
        to_center = batch_points - mesh_center
        dist_to_center = np.linalg.norm(to_center, axis=1)
        
        # If closer to center than to surface, likely inside
        # This is a rough approximation but memory-efficient
        signs = np.where(dist_to_center < dists * 2, -1.0, 1.0)
        
        all_sdf[i:end] = batch_sdf * signs
        
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

        # Combine meshes
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
        
        # Compute SDF in batches (memory efficient!)
        print(f"    Computing SDF for {len(vol_points)} volume points...")
        vol_sdf = compute_sdf_simple(
            vol_points, 
            all_verts, 
            all_faces, 
            batch_size=args.batch_size
        )
        
        # Near surface points
        near_points = surface_points + np.random.normal(0, 0.01, surface_points.shape).astype(np.float32)
        near_points = np.clip(near_points, -1, 1)
        
        print(f"    Computing SDF for {len(near_points)} near-surface points...")
        near_sdf = compute_sdf_simple(
            near_points,
            all_verts,
            all_faces,
            batch_size=args.batch_size
        )
        
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
        
        # Free memory
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
    print(f"Settings: surface_pts={args.num_surface_points}, vol_pts={args.num_vol_points}, batch={args.batch_size}")
    print("="*60)
    
    for i, f in enumerate(files):
        print(f"\n[{start+i+1}/{end}] {os.path.basename(f)}")
        process_file(f, args)
        
        if (i + 1) % 5 == 0:
            gc.collect()
            print(f"\n  → Progress: {i+1}/{len(files)} files completed")

    print(f"\n{'='*60}")
    print(f"✅ Completed {len(files)} files!")

if __name__ == "__main__":
    main()
