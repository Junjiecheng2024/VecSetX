import os
import glob
import argparse
import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
from scipy.spatial import cKDTree
import gc
from multiprocessing import Pool, cpu_count
from functools import partial

# -----------------------------------------------------------------------------
# Configuration & Deps
# -----------------------------------------------------------------------------
try:
    import mesh_to_sdf
    MESH_TO_SDF_AVAILABLE = True
except ImportError:
    MESH_TO_SDF_AVAILABLE = False

def get_args():
    parser = argparse.ArgumentParser(description="Multi-Class Hybrid Final: Heuristic Vol + Accurate Near (OPTIMIZED)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_surface_points", type=int, default=50000)
    parser.add_argument("--num_vol_points", type=int, default=50000)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100000, 
                       help="Batch size for SDF computation. Use full size for maximum speed on clusters.")
    parser.add_argument("--vol_threshold", type=float, default=0.5, 
                       help="Heuristic threshold for vol_sdf inside/outside判断. Lower=more inside. 0.5→~50/50 (recommended)")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--n_workers", type=int, default=cpu_count(),
                       help="Number of parallel workers for per-class SDF computation. 0=auto (use all CPUs)")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Heuristic Logic (Fast, for Vol SDF & Labels)
# -----------------------------------------------------------------------------
def compute_heuristic_sdf_single_batch(query_points, mesh_vertices, threshold=0.5):
    """
    Computes SDF for a SINGLE batch (no batching inside).
    Optimized for large memory systems.
    """
    # KDTree for distance
    tree = cKDTree(mesh_vertices)
    
    # Centroid for Heuristic
    mesh_center = mesh_vertices.mean(axis=0)
    
    # 1. Dist to Surface
    dists, _ = tree.query(query_points, k=1)
    
    # 2. Dist to Center
    to_center = query_points - mesh_center
    dist_to_center = np.linalg.norm(to_center, axis=1)
    
    # 3. Depth Ratio
    depth_ratio = dists / (dist_to_center + 1e-8)
    
    # 4. Sign determination
    is_inside = depth_ratio > threshold
    signs = np.where(is_inside, -1.0, 1.0)
    
    return (dists * signs).astype(np.float32)

def compute_sdf_for_class(args_tuple):
    """
    Wrapper for parallel processing.
    args_tuple: (query_points, mesh_vertices, threshold, class_idx)
    """
    query_points, mesh_vertices, threshold, class_idx = args_tuple
    sdf = compute_heuristic_sdf_single_batch(query_points, mesh_vertices, threshold)
    return class_idx, sdf

# -----------------------------------------------------------------------------
# MeshToSDF Logic (Slow but Accurate, for Near SDF)
# -----------------------------------------------------------------------------
def compute_high_quality_sdf(query_points, mesh):
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

# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------
def process_file(file_path, args):
    try:
        if not MESH_TO_SDF_AVAILABLE:
            print("    ✗ Skipped: mesh_to_sdf missing")
            return
            
        # 1. Load Data
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # 2. Extract Meshes (Class 1..10)
        meshes = {}
        total_surface_area = 0
        failed_classes = []
        
        for c in range(1, args.classes + 1):
            mask = (data == c)
            if np.sum(mask) == 0:
                print(f"    ⚠️ Class {c}: No voxels found")
                failed_classes.append(c)
                continue
            try:
                verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                meshes[c] = mesh
                total_surface_area += mesh.area
            except Exception as e:
                print(f"    ⚠️ Class {c}: Marching cubes failed - {e}")
                failed_classes.append(c)
                continue
        
        if failed_classes:
            print(f"    Missing classes: {failed_classes}")
        
        if total_surface_area == 0:
            print("    Error: No surface found")
            return

        # 3. Sample Surface & Normals (for Near Points)
        surface_points_list = []
        surface_labels_list = []
        combined_verts = []
        combined_faces = []
        v_offset = 0

        for c, mesh in meshes.items():
            # Sample for near points
            n_samples = max(1, int(args.num_surface_points * (mesh.area / total_surface_area)))
            p, _ = trimesh.sample.sample_surface(mesh, n_samples)
            surface_points_list.append(p)
            
            # Record label
            l = np.zeros((n_samples, args.classes), dtype=np.float32)
            l[:, c-1] = 1.0
            surface_labels_list.append(l)

            # Accumulate for Union Mesh
            combined_verts.append(mesh.vertices)
            combined_faces.append(mesh.faces + v_offset)
            v_offset += len(mesh.vertices)

        surface_points = np.concatenate(surface_points_list, axis=0)
        surface_labels = np.concatenate(surface_labels_list, axis=0)
        
        # 4. Normalize
        shifts = (surface_points.max(axis=0) + surface_points.min(axis=0)) / 2
        surface_points = surface_points - shifts
        max_dist = np.max(np.linalg.norm(surface_points, axis=1))
        scale_factor = 1.0 / max_dist if max_dist > 0 else 1.0
        
        surface_points *= scale_factor
        
        # Scale all class meshes (for Heuristic)
        scaled_meshes = {}
        for c, m in meshes.items():
            v = (m.vertices - shifts) * scale_factor
            scaled_meshes[c] = v  # Only store vertices
        
        # Scale Union Mesh (for mesh_to_sdf)
        all_v = (np.concatenate(combined_verts) - shifts) * scale_factor
        all_f = np.concatenate(combined_faces)
        union_mesh = trimesh.Trimesh(vertices=all_v, faces=all_f)
        
        # ---------------------------------------------------------------------
        # GENERATE POINTS
        # ---------------------------------------------------------------------
        vol_points = np.random.uniform(-1, 1, (args.num_vol_points, 3)).astype(np.float32)
        near_points = surface_points + np.random.normal(0, 0.01, surface_points.shape).astype(np.float32)
        near_points = np.clip(near_points, -1, 1)

        # ---------------------------------------------------------------------
        # COMPUTE: VOLUME POINTS (Pure Heuristic) - PARALLEL
        # ---------------------------------------------------------------------
        print(f"    Computing Vol SDF (Parallel, {len(scaled_meshes)} classes)...")
        
        # Prepare arguments for parallel processing
        n_workers = args.n_workers if args.n_workers > 0 else cpu_count()
        tasks = [
            (vol_points, verts, args.vol_threshold, c-1)
            for c, verts in scaled_meshes.items()
        ]
        
        # Parallel computation
        vol_sdf_all = np.full((len(vol_points), args.classes), 99.0, dtype=np.float32)
        
        with Pool(processes=min(n_workers, len(tasks))) as pool:
            results = pool.map(compute_sdf_for_class, tasks)
        
        for class_idx, sdf in results:
            vol_sdf_all[:, class_idx] = sdf
        
        # Combine: Union SDF is minimum across all classes
        vol_sdf = np.min(vol_sdf_all, axis=1)
        vol_labels = np.zeros(len(vol_points), dtype=np.int8)
        
        # CRITICAL FIX: Only assign labels where Union SDF says "inside"
        is_inside_vol = vol_sdf < 0
        
        # For inside points, find the class with MOST NEGATIVE (deepest inside) SDF
        # This ensures the assigned class also thinks the point is inside
        for i in np.where(is_inside_vol)[0]:
            # Get SDF values for this point across all classes
            class_sdfs = vol_sdf_all[i, :]
            # Find classes that consider this point inside (SDF < 0)
            inside_classes = np.where(class_sdfs < 0)[0]
            if len(inside_classes) > 0:
                # Assign label of the class with most negative SDF
                deepest_class = inside_classes[np.argmin(class_sdfs[inside_classes])]
                vol_labels[i] = deepest_class + 1
            # else: leave as 0 (background) if no class thinks it's inside
        
        # ---------------------------------------------------------------------
        # COMPUTE: NEAR POINTS (Hybrid)
        # ---------------------------------------------------------------------
        print(f"    Computing Near SDF (mesh_to_sdf)...")
        
        # A. Geometry (High Quality)
        near_sdf_union = compute_high_quality_sdf(near_points, union_mesh)
        
        # B. Semantics (Heuristic Check) - PARALLEL
        print(f"    Computing Near Labels (Parallel)...")
        tasks_near = [
            (near_points, verts, args.vol_threshold, c-1)
            for c, verts in scaled_meshes.items()
        ]
        
        near_sdf_heur_all = np.full((len(near_points), args.classes), 99.0, dtype=np.float32)
        
        with Pool(processes=min(n_workers, len(tasks_near))) as pool:
            results_near = pool.map(compute_sdf_for_class, tasks_near)
        
        for class_idx, sdf in results_near:
            near_sdf_heur_all[:, class_idx] = sdf
        
        # Heuristic Labels - FIXED for consistency
        near_labels = np.zeros(len(near_points), dtype=np.int8)
        
        # C. Fusion: Use mesh_to_sdf geometry, assign labels only where inside
        near_sdf = near_sdf_union
        is_inside_near = near_sdf < 0
        
        # For inside points, find the class with most negative heuristic SDF
        for i in np.where(is_inside_near)[0]:
            class_sdfs = near_sdf_heur_all[i, :]
            inside_classes = np.where(class_sdfs < 0)[0]
            if len(inside_classes) > 0:
                deepest_class = inside_classes[np.argmin(class_sdfs[inside_classes])]
                near_labels[i] = deepest_class + 1
            # else: leave as 0 if no class thinks it's inside
        
        # ---------------------------------------------------------------------
        # SAVE
        # ---------------------------------------------------------------------
        filename = os.path.basename(file_path).replace('.nii.gz', '.npz')
        save_path = os.path.join(args.output_dir, filename)
        
        vol_sdf = vol_sdf.reshape(-1, 1)
        near_sdf = near_sdf.reshape(-1, 1)
        
        # Log ratios
        vp = (vol_sdf > 0).mean()
        np_pos = (near_sdf > 0).mean()
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
                 
        del meshes, scaled_meshes, union_mesh
        del vol_points, vol_sdf, vol_labels, vol_sdf_all
        del near_points, near_sdf, near_labels, near_sdf_heur_all
        gc.collect()
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    args = get_args()
    if not MESH_TO_SDF_AVAILABLE:
        print("ERROR: mesh_to_sdf required.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.nii.gz")))
    
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(files)
    files = files[start:end]
    
    n_workers = args.n_workers if args.n_workers > 0 else cpu_count()
    
    print(f"Processing {len(files)} files. Hybrid Final (Multi-Class) OPTIMIZED.")
    print(f"Parallel Workers: {n_workers} (per file)")
    print(f"Batch Size: {args.batch_size}")
    print("="*60)
    
    for i, f in enumerate(files):
        print(f"[{i+1}/{len(files)}] {os.path.basename(f)}")
        process_file(f, args)
        if (i+1) % 10 == 0: gc.collect()
        
    print("Done.")

if __name__ == "__main__":
    main()
