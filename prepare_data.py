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
    parser.add_argument("--file_workers", type=int, default=1,
                       help="Number of files to process in parallel. Set to 4-8 for faster generation on multi-core systems.")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Accurate Logic (MeshToSDF for All)
# -----------------------------------------------------------------------------
def compute_accurate_sdf_single_batch(query_points, mesh):
    """
    Computes accurate SDF using mesh_to_sdf.
    """
    if not MESH_TO_SDF_AVAILABLE:
        raise ImportError("mesh_to_sdf required")
    
    # mesh_to_sdf returns negative for inside, positive for outside
    sdf = mesh_to_sdf.mesh_to_sdf(
        mesh, 
        query_points, 
        surface_point_method='sample',
        sign_method='normal', 
        scan_count=100,
        scan_resolution=400,
        sample_point_count=10000,
        normal_sample_count=100
    )
    return sdf.astype(np.float32)

def compute_sdf_for_class(args_tuple):
    """
    Wrapper for parallel processing.
    args_tuple: (query_points, mesh_vertices, mesh_faces, class_idx)
    Note: mesh_to_sdf needs a Trimesh object. 
    We pass vertices/faces to reconstruct it inside the worker to avoid pickling issues?
    Or just pass the mesh if pickling works (usually fine with fork).
    Let's try passing (query_points, mesh, class_idx).
    """
    query_points, mesh_data, class_idx = args_tuple
    
    # If mesh_data is (verts, faces), reconstruct
    if isinstance(mesh_data, tuple):
        verts, faces = mesh_data
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    else:
        mesh = mesh_data
        
    sdf = compute_accurate_sdf_single_batch(query_points, mesh)
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
        # GENERATE POINTS - STRATIFIED SAMPLING IN NORMALIZED SPACE
        # ---------------------------------------------------------------------
        # Instead of sampling in voxel space (which causes coordinate mismatch),
        # we do stratified sampling in the SAME normalized space where meshes live
        
        # Strategy:
        # - 35% uniform random (background biased)
        # - 65% sampled near/inside structures (using mesh bounds)
        
        n_bg = int(args.num_vol_points * 0.35)
        n_struct = args.num_vol_points - n_bg
        
        # Background points: pure uniform random in [-1, 1]^3
        vol_points_bg = np.random.uniform(-1, 1, (n_bg, 3)).astype(np.float32)
        
        # Structure-biased points: sample within the bounding box of union mesh,
        # then add some noise to also sample slightly outside
        bounds = union_mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        bbox_min = bounds[0]
        bbox_max = bounds[1]
        
        # Expand bbox slightly (10%) to also sample near-structure regions
        bbox_range = bbox_max - bbox_min
        bbox_min_expanded = bbox_min - 0.1 * bbox_range
        bbox_max_expanded = bbox_max + 0.1 * bbox_range
        
        # Clip to [-1, 1] bounds
        bbox_min_expanded = np.maximum(bbox_min_expanded, -1.0)
        bbox_max_expanded = np.minimum(bbox_max_expanded, 1.0)
        
        # Sample uniformly within expanded bbox
        vol_points_struct = np.random.uniform(
            bbox_min_expanded, 
            bbox_max_expanded, 
            (n_struct, 3)
        ).astype(np.float32)
        
        # Combine
        vol_points = np.vstack([vol_points_bg, vol_points_struct])
        
        print(f"    Stratified sampling (normalized space): {n_bg} bg + {n_struct} struct = {len(vol_points)} total")
        near_points = surface_points + np.random.normal(0, 0.01, surface_points.shape).astype(np.float32)
        near_points = np.clip(near_points, -1, 1)

        # ---------------------------------------------------------------------
        # COMPUTE: VOLUME POINTS (Accurate) - PARALLEL
        # ---------------------------------------------------------------------
        print(f"    Computing Vol SDF (Accurate, {len(meshes)} classes)...")
        
        # CRITICAL: When file_workers > 1, we can't use Pool inside Pool (daemon issue)
        # So we check if n_workers should be disabled
        n_workers = args.n_workers if args.n_workers > 0 else cpu_count()
        
        # If this is being called from a parallel file worker, disable internal parallelization
        use_parallel = (n_workers > 1) and (args.file_workers <= 1)
        
        # Calculate shifts/scale again or use pre-calculated? 
        # Wait, meshes dict has ORIGINAL meshes. 
        # scaled_meshes had only VERTICES.
        # We need FULL MESHES (verts+faces) for mesh_to_sdf.
        # Let's create `scaled_full_meshes`
        scaled_full_meshes = {}
        for c, m in meshes.items():
            v = (m.vertices - shifts) * scale_factor
            scaled_full_meshes[c] = (v, m.faces) # Pass as tuple
        
        if use_parallel:
            # Prepare arguments for parallel processing
            tasks = [
                (vol_points, (verts, faces), c-1)
                for c, (verts, faces) in scaled_full_meshes.items()
            ]
            
            # Parallel computation
            vol_sdf_all = np.full((len(vol_points), args.classes), 99.0, dtype=np.float32)
            
            with Pool(processes=min(n_workers, len(tasks))) as pool:
                results = pool.map(compute_sdf_for_class, tasks)
            
            for class_idx, sdf in results:
                vol_sdf_all[:, class_idx] = sdf
        else:
            # Sequential computation (when file_workers > 1 or n_workers == 1)
            vol_sdf_all = np.full((len(vol_points), args.classes), 99.0, dtype=np.float32)
            for c, (verts, faces) in scaled_full_meshes.items():
                # Reconstruct mesh locally
                mesh_temp = trimesh.Trimesh(vertices=verts, faces=faces)
                sdf = compute_accurate_sdf_single_batch(vol_points, mesh_temp)
                vol_sdf_all[:, c-1] = sdf
        
        # Combine: Union SDF is minimum across all classes
        vol_sdf = np.min(vol_sdf_all, axis=1)
        
        # Assign labels based on SDF (mathematically consistent)
        vol_labels = np.zeros(len(vol_points), dtype=np.int8)
        
        # Only assign labels where Union SDF says "inside"
        is_inside_vol = vol_sdf < 0
        
        # For inside points, find the class with MOST NEGATIVE (deepest inside) SDF
        for i in np.where(is_inside_vol)[0]:
            class_sdfs = vol_sdf_all[i, :]
            inside_classes = np.where(class_sdfs < 0)[0]
            if len(inside_classes) > 0:
                deepest_class = inside_classes[np.argmin(class_sdfs[inside_classes])]
                vol_labels[i] = deepest_class + 1
            # else: leave as 0 if no class thinks it's inside
        
        print(f"    Vol labels: {len(np.unique(vol_labels))} classes present")
        
        # ---------------------------------------------------------------------
        # COMPUTE: NEAR POINTS (Hybrid)
        # ---------------------------------------------------------------------
        print(f"    Computing Near SDF (mesh_to_sdf)...")
        
        # A. Geometry (High Quality)
        near_sdf_union = compute_high_quality_sdf(near_points, union_mesh)
        
        # B. Semantics (Accurate Check) - PARALLEL (if allowed)
        print(f"    Computing Near Labels (Accurate)...")
        
        if use_parallel:
            tasks_near = [
                (near_points, (verts, faces), c-1)
                for c, (verts, faces) in scaled_full_meshes.items()
            ]
            
            near_sdf_class_all = np.full((len(near_points), args.classes), 99.0, dtype=np.float32)
            
            with Pool(processes=min(n_workers, len(tasks_near))) as pool:
                results_near = pool.map(compute_sdf_for_class, tasks_near)
            
            for class_idx, sdf in results_near:
                near_sdf_class_all[:, class_idx] = sdf
        else:
            # Sequential
            near_sdf_class_all = np.full((len(near_points), args.classes), 99.0, dtype=np.float32)
            for c, (verts, faces) in scaled_full_meshes.items():
                mesh_temp = trimesh.Trimesh(vertices=verts, faces=faces)
                sdf = compute_accurate_sdf_single_batch(near_points, mesh_temp)
                near_sdf_class_all[:, c-1] = sdf
        
        # Heuristic Labels - FIXED for consistency
        near_labels = np.zeros(len(near_points), dtype=np.int8)
        
        # C. Fusion: Use mesh_to_sdf geometry, assign labels only where inside
        near_sdf = near_sdf_union
        is_inside_near = near_sdf < 0
        
        # For inside points, find the class with most negative SDF
        for i in np.where(is_inside_near)[0]:
            class_sdfs = near_sdf_class_all[i, :]
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
        
        return os.path.basename(file_path)  # CRITICAL: Must return for parallel pool
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None  # Return None on failure

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
    file_workers = args.file_workers
    
    print(f"Processing {len(files)} files. Hybrid Final (Multi-Class) OPTIMIZED.")
    print(f"Parallel Workers: {n_workers} (per file)")
    print(f"File Workers: {file_workers} (parallel files)")
    print(f"Batch Size: {args.batch_size}")
    print("="*60)
    
    if file_workers > 1:
        # Multi-file parallel processing
        print(f"Using multi-file parallel mode with {file_workers} workers")
        
        # Create a partial function with args
        process_func = partial(process_file, args=args)
        
        # Process files in parallel
        with Pool(processes=file_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(process_func, files)):
                if (i+1) % 10 == 0:
                    print(f"Progress: {i+1}/{len(files)} files completed")
                    gc.collect()
    else:
        # Sequential processing (original behavior)
        for i, f in enumerate(files):
            print(f"[{i+1}/{len(files)}] {os.path.basename(f)}")
            process_file(f, args)
            if (i+1) % 10 == 0: 
                gc.collect()
        
    print("Done.")

if __name__ == "__main__":
    main()
