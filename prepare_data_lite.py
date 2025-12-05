import os
import glob
import argparse
import numpy as np
import nibabel as nib
import trimesh
import torch
from skimage import measure
from scipy.spatial import cKDTree
import gc  # For garbage collection

def get_args():
    parser = argparse.ArgumentParser(description="Convert .nii.gz masks to .npz for VecSetX (Batch Processing)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .nii.gz files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npz files")
    parser.add_argument("--num_surface_points", type=int, default=50000, help="Total number of surface points to sample")
    parser.add_argument("--num_vol_points", type=int, default=50000, help="Total number of volume points to sample")
    parser.add_argument("--classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (for resuming)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (None = all files)")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size for SDF computation")
    return parser.parse_args()

def compute_sdf_simple(query_points, mesh_vertices, mesh_faces, batch_size=5000):
    """
    Memory-efficient SDF computation using batching
    Uses heuristic method (no rtree required)
    """
    n_points = len(query_points)
    all_sdf = np.zeros(n_points, dtype=np.float32)
    
    # Build KDTree once
    tree = cKDTree(mesh_vertices)
    
    # Mesh bounds for heuristic
    mesh_center = mesh_vertices.mean(axis=0)
    
    # Process in batches
    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        batch_points = query_points[i:end]
        
        # Find nearest vertices
        dists, idxs = tree.query(batch_points, k=1)
        
        # Heuristic: use distance to mesh center
        to_center = batch_points - mesh_center
        dist_to_center = np.linalg.norm(to_center, axis=1)
        
        # If closer to center than to surface, likely inside
        # Adjust threshold for better accuracy
        signs = np.where(dist_to_center < dists * 1.5, -1.0, 1.0)
        
        batch_sdf = dists.astype(np.float32)
        all_sdf[i:end] = batch_sdf * signs
        
    return all_sdf

def process_file(file_path, args):
    try:
        # Load NIfTI file
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # 1. Extract Surface Points and Labels
        surface_points_list = []
        surface_labels_list = []
        
        total_surface_area = 0
        meshes = {}
        
        for c in range(1, args.classes + 1):
            mask = (data == c)
            if np.sum(mask) == 0:
                continue
                
            # Marching Cubes to get mesh
            try:
                verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                meshes[c] = mesh
                total_surface_area += mesh.area
            except Exception as e:
                print(f"    Warning: Failed to extract mesh for class {c}: {e}")
                continue

        if total_surface_area == 0:
            print(f"    Error: No surface found")
            return

        # Sample points proportional to area
        for c, mesh in meshes.items():
            n_samples = int(args.num_surface_points * (mesh.area / total_surface_area))
            if n_samples == 0: 
                continue
            
            points, _ = trimesh.sample.sample_surface(mesh, n_samples)
            surface_points_list.append(points)
            
            # Create one-hot label
            label = np.zeros((n_samples, args.classes), dtype=np.float32)
            label[:, c-1] = 1.0
            surface_labels_list.append(label)
            
        if not surface_points_list:
            return

        surface_points = np.concatenate(surface_points_list, axis=0)
        surface_labels = np.concatenate(surface_labels_list, axis=0)
        
        # Normalize surface points to [-1, 1]
        shifts = (surface_points.max(axis=0) + surface_points.min(axis=0)) / 2
        surface_points = surface_points - shifts
        distances = np.linalg.norm(surface_points, axis=1)
        max_dist = np.max(distances)
        if max_dist > 0:
            scale_factor = 1 / max_dist
            surface_points *= scale_factor
        else:
            scale_factor = 1.0

        # 2. Generate Volume Points and SDF
        vol_points = np.random.uniform(-1, 1, (args.num_vol_points, 3)).astype(np.float32)
        
        # Combine all meshes for SDF calculation
        combined_verts = []
        combined_faces = []
        vertex_offset = 0
        
        for c, mesh in meshes.items():
            v = mesh.vertices
            # Normalize vertices same as points
            v = v - shifts
            v *= scale_factor
            
            combined_verts.append(v)
            combined_faces.append(mesh.faces + vertex_offset)
            vertex_offset += len(v)
            
        all_verts = np.concatenate(combined_verts)
        all_faces = np.concatenate(combined_faces)
        
        # Compute SDF
        print(f"    Computing vol_sdf ({len(vol_points)} points, batch_size={args.batch_size})...")
        vol_sdf = compute_sdf_simple(
            vol_points, 
            all_verts, 
            all_faces, 
            batch_size=args.batch_size
        )
        
        # Near surface points
        near_points = surface_points + np.random.normal(0, 0.01, surface_points.shape).astype(np.float32)
        near_points = np.clip(near_points, -1, 1)
        
        print(f"    Computing near_sdf ({len(near_points)} points, batch_size={args.batch_size})...")
        near_sdf = compute_sdf_simple(
            near_points,
            all_verts,
            all_faces,
            batch_size=args.batch_size
        )
        
        # Reshape SDF
        vol_sdf = vol_sdf.reshape(-1, 1)
        near_sdf = near_sdf.reshape(-1, 1)
        
        # 3. Save
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
        
        # Explicitly free memory
        del meshes, surface_points, vol_points, near_points
        del vol_sdf, near_sdf, all_verts, all_faces
        gc.collect()

    except Exception as e:
        print(f"    ✗ Failed to process {file_path}: {e}")

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.nii.gz")))
    print(f"Found {len(files)} files.")
    
    # Apply start and end index
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(files)
    files = files[start:end]
    
    print(f"Processing files {start} to {end-1} ({len(files)} files)")
    
    for i, f in enumerate(files):
        print(f"\n[{start + i + 1}/{end}] Processing: {os.path.basename(f)}")
        process_file(f, args)
        
        # Force garbage collection every 10 files
        if (i + 1) % 10 == 0:
            gc.collect()
            print(f"  → Processed {i + 1}/{len(files)} files, memory freed")

    print(f"\n✅ Completed processing {len(files)} files!")

if __name__ == "__main__":
    main()
