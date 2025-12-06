import os
import glob
import argparse
import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
from scipy.spatial import cKDTree
import gc

try:
    import mesh_to_sdf
    MESH_TO_SDF_AVAILABLE = True
except ImportError:
    MESH_TO_SDF_AVAILABLE = False

def get_args():
    parser = argparse.ArgumentParser(description="Multi-Class Data Generation (SDF + Labels)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_surface_points", type=int, default=50000)
    parser.add_argument("--num_vol_points", type=int, default=50000)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--vol_threshold", type=float, default=0.33, 
                       help="Threshold for depth ratio. 0.33 is optimized for ~50/50 balance.")
    return parser.parse_args()

def is_inside_mesh(query_points, mesh_vertices, threshold=0.33, batch_size=5000):
    """
    Check if points are inside the mesh using Depth Ratio Heuristic.
    Returns boolean array.
    """
    n_points = len(query_points)
    all_is_inside = np.zeros(n_points, dtype=bool)
    
    tree = cKDTree(mesh_vertices)
    mesh_center = mesh_vertices.mean(axis=0)
    
    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        batch = query_points[i:end]
        
        # Distance to nearest surface point
        dists, _ = tree.query(batch, k=1)
        
        # Distance to mesh center
        to_center = batch - mesh_center
        dist_to_center = np.linalg.norm(to_center, axis=1)
        
        # Depth ratio
        depth_ratio = dists / (dist_to_center + 1e-8)
        
        # Check inside
        all_is_inside[i:end] = depth_ratio > threshold
        
    return all_is_inside

def compute_vol_sdf_truly_tunable(query_points, mesh_vertices, threshold=0.33, batch_size=5000):
    """
    Compute signed distance for the Union Mesh.
    Reuse the same tree/logic but return float SDF.
    """
    n_points = len(query_points)
    all_sdf = np.zeros(n_points, dtype=np.float32)
    
    tree = cKDTree(mesh_vertices)
    mesh_center = mesh_vertices.mean(axis=0)
    
    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        batch = query_points[i:end]
        
        dists, _ = tree.query(batch, k=1)
        to_center = batch - mesh_center
        dist_to_center = np.linalg.norm(to_center, axis=1)
        depth_ratio = dists / (dist_to_center + 1e-8)
        
        is_inside = depth_ratio > threshold
        signs = np.where(is_inside, -1.0, 1.0)
        all_sdf[i:end] = dists.astype(np.float32) * signs
        
    return all_sdf

def compute_sdf_mesh_to_sdf(query_points, mesh):
    """mesh_to_sdf for near_sdf (perfect accuracy)"""
    if not MESH_TO_SDF_AVAILABLE:
        raise ImportError("mesh_to_sdf required")
    
    sdf_values = mesh_to_sdf.mesh_to_sdf(
        mesh, 
        query_points,
        surface_point_method='sample',
        sign_method='normal',
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
        surface_points_list = []
        surface_labels_list = []
        near_points_list = []
        near_labels_list = []
        
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

        # 1. Process Surface & Near Points (Per Class)
        for c, mesh in meshes.items():
            # Calculate proportion
            ratio = mesh.area / total_surface_area
            n_surf = max(1, int(args.num_surface_points * ratio))
            
            # --- Surface Points ---
            points, _ = trimesh.sample.sample_surface(mesh, n_surf)
            surface_points_list.append(points)
            
            # One-hot label for input
            label = np.zeros((n_surf, args.classes), dtype=np.float32)
            label[:, c-1] = 1.0
            surface_labels_list.append(label)
            
            # --- Near Points ---
            # Sample near points around this specific mesh
            # We want near points to have the label of the mesh they are near to?
            # Actually, near points might be inside another mesh if they are close.
            # But for simplicity, we assign the label of the source mesh.
            # Or we can verify later. Let's start with source label.
            # Wait, near points should serve as training data for the 10-class segmentation.
            # If a point is near Left Atrium, it is likely Left Atrium or Background (or neighbor).
            # Let's generate points and then decide label globally or locally.
            # Strategy: Generate points -> global label check?
            # Global check is expensive.
            # Local assumption: near points of Mesh C are Class C (if inside) or 0 (if outside).
            # BUT, we need precise labels.
            # Let's generate near points here, but assign labels later (or keep track).
            # Actually, `near_points` are usually very close (sigma=0.01).
            # Let's collect them and perform Global Labeling on them alongside Vol points!
            # That ensures consistency.
            
        surface_points = np.concatenate(surface_points_list, axis=0)
        surface_labels = np.concatenate(surface_labels_list, axis=0)
        
        # Normalize
        shifts = (surface_points.max(axis=0) + surface_points.min(axis=0)) / 2
        surface_points = surface_points - shifts
        max_dist = np.max(np.linalg.norm(surface_points, axis=1))
        scale_factor = 1 / max_dist if max_dist > 0 else 1.0
        surface_points *= scale_factor
        
        # Apply normalization to all meshes
        combined_verts = []
        combined_faces = []
        vertex_offset = 0
        
        scaled_meshes = {} # Store for labeling
        
        for c, mesh in meshes.items():
            v = (mesh.vertices - shifts) * scale_factor
            scaled_mesh = trimesh.Trimesh(vertices=v, faces=mesh.faces)
            scaled_meshes[c] = scaled_mesh
            
            combined_verts.append(v)
            combined_faces.append(mesh.faces + vertex_offset)
            vertex_offset += len(v)
            
        all_verts = np.concatenate(combined_verts)
        all_faces = np.concatenate(combined_faces)
        combined_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
        
        # 2. Generate Query Points
        
        # Volume Points
        vol_points = np.random.uniform(-1, 1, (args.num_vol_points, 3)).astype(np.float32)
        
        # Near Points (regenerate globally around combined mesh to match distribution?)
        # Or use the per-mesh surfaces?
        # Let's use the normalized surface points we already have to generate near points
        near_points = surface_points + np.random.normal(0, 0.01, surface_points.shape).astype(np.float32)
        near_points = np.clip(near_points, -1, 1)
        
        # 3. Compute Union SDF (Geometry)
        print(f"    Computing Union SDF...")
        vol_sdf = compute_vol_sdf_truly_tunable(vol_points, all_verts, threshold=args.vol_threshold, batch_size=args.batch_size)
        near_sdf = compute_sdf_mesh_to_sdf(near_points, combined_mesh)
        
        # 4. Compute Semantic Labels (Class 1-10 or 0)
        print(f"    Computing Semantic Labels...")
        
        vol_labels = np.zeros(len(vol_points), dtype=np.int8)
        near_labels = np.zeros(len(near_points), dtype=np.int8)
        
        # For each class, check if points are inside
        # Note: In case of overlap, the last one wins. (Medical data usually non-overlapping)
        # We use the same heuristic for consistency with vol_sdf
        
        for c, mesh in scaled_meshes.items():
            # Check Volume Points
            # Use heuristic for volume points (consistent with vol_sdf)
            is_inside_vol = is_inside_mesh(vol_points, mesh.vertices, threshold=args.vol_threshold, batch_size=args.batch_size)
            vol_labels[is_inside_vol] = c
            
            # Check Near Points
            # For near points, we want high precision. Use mesh_to_sdf sign check?
            # Or use mesh.contains? 
            # Given we used mesh_to_sdf for near_sdf, we should use it for labels too?
            # mesh_to_sdf is slow if we do it 10 times.
            # BUT, we know `near_points` were generated from `surface_points`.
            # A point generated from Surface C is very likely Class C (or 0).
            # But it could be Class D if D is adjacent.
            # Let's use the heuristic for now to be fast and consistent. 
            # Or better: strictly check "inside" for labels.
            
            # Let's use the same heuristic. It works well for "inside/outside".
            is_inside_near = is_inside_mesh(near_points, mesh.vertices, threshold=args.vol_threshold, batch_size=args.batch_size)
            near_labels[is_inside_near] = c

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
                 vol_labels=vol_labels,  # New
                 near_points=near_points.astype(np.float32),
                 near_sdf=near_sdf.astype(np.float32),
                 near_labels=near_labels # New
                 )
        
        print(f"    ✓ Saved {filename} (Labels included)")
        
        del combined_mesh, meshes, scaled_meshes
        del vol_points, vol_sdf, vol_labels
        del near_points, near_sdf, near_labels
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
    print(f"Multi-Class Mode enabled.")
    
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
