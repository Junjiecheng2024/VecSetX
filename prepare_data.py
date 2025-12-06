import os
import glob
import argparse
import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
from scipy.spatial import cKDTree
import gc

# -----------------------------------------------------------------------------
# Configuration & Deps
# -----------------------------------------------------------------------------
try:
    import mesh_to_sdf
    MESH_TO_SDF_AVAILABLE = True
except ImportError:
    MESH_TO_SDF_AVAILABLE = False

def get_args():
    parser = argparse.ArgumentParser(description="Multi-Class Hybrid Final: Heuristic Vol + Accurate Near")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_surface_points", type=int, default=50000)
    parser.add_argument("--num_vol_points", type=int, default=50000)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--vol_threshold", type=float, default=0.5, 
                       help="Heuristic threshold for vol_sdf inside/outside判断. Lower=more inside. 0.5→~50/50 (recommended)")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Heuristic Logic (Fast, for Vol SDF & Labels)
# -----------------------------------------------------------------------------
def compute_heuristic_sdf(query_points, mesh_vertices, threshold=0.33, batch_size=5000):
    """
    Computes SDF approximation using Depth Ratio Heuristic.
    Returns: values (N,) where values < 0 means inside.
    """
    n_points = len(query_points)
    all_sdf = np.zeros(n_points, dtype=np.float32)
    
    # KDTree for distance
    tree = cKDTree(mesh_vertices)
    
    # Centroid for Heuristic
    mesh_center = mesh_vertices.mean(axis=0)
    
    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        batch = query_points[i:end]
        
        # 1. Dist to Surface
        dists, _ = tree.query(batch, k=1)
        
        # 2. Dist to Center
        to_center = batch - mesh_center
        dist_to_center = np.linalg.norm(to_center, axis=1)
        
        # 3. Depth Ratio: measures relative position
        # depth_ratio = dist_to_surface / dist_to_center
        # High ratio (>threshold): surface far, center close → INSIDE
        # Low ratio (<threshold): surface close, center far → OUTSIDE
        depth_ratio = dists / (dist_to_center + 1e-8)
        
        # 4. Sign determination
        # threshold=1.2 gives ~50% positive (recommended)
        is_inside = depth_ratio > threshold
        signs = np.where(is_inside, -1.0, 1.0)
        
        all_sdf[i:end] = dists.astype(np.float32) * signs
        
    return all_sdf

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
        valid = True
        
        for c in range(1, args.classes + 1):
            mask = (data == c)
            if np.sum(mask) == 0:
                continue
            try:
                verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                meshes[c] = mesh
                total_surface_area += mesh.area
            except:
                continue
        
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
            
            # Record label (for potential use, though checking consistency later)
            l = np.zeros((n_samples, args.classes), dtype=np.float32)
            l[:, c-1] = 1.0
            surface_labels_list.append(l)

            # Accumulate for Union Mesh (High Quality SDF input)
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
        
        surface_points *= scale_factor # Normalized surface points
        
        # Scale all class meshes (for Heuristic)
        scaled_meshes = {}
        for c, m in meshes.items():
            v = (m.vertices - shifts) * scale_factor
            scaled_meshes[c] = (v, m.faces) # Store verts/faces
        
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
        # COMPUTE: VOLUME POINTS (Pure Heuristic)
        # Fast, Per-Class check
        # ---------------------------------------------------------------------
        # print("    Computing Vol SDF (Heuristic)...")
        
        # Matrix: (N, 10)
        vol_sdf_all = np.full((len(vol_points), args.classes), 99.0, dtype=np.float32)
        
        for c in scaled_meshes.keys():
            verts, _ = scaled_meshes[c]
            vol_sdf_all[:, c-1] = compute_heuristic_sdf(
                vol_points, verts, threshold=args.vol_threshold, batch_size=args.batch_size
            )
            
        # Combine
        # Union SDF = min(Class SDFs)
        vol_sdf = np.min(vol_sdf_all, axis=1) # (N,)
        
        # Labels = argmin(Class SDFs) + 1 (if Inside Union)
        # Note: If point is outside ALL classes, vol_sdf > 0.
        vol_labels = np.zeros(len(vol_points), dtype=np.int8)
        min_indices = np.argmin(vol_sdf_all, axis=1)
        
        is_inside_vol = vol_sdf < 0
        vol_labels[is_inside_vol] = min_indices[is_inside_vol] + 1
        
        # ---------------------------------------------------------------------
        # COMPUTE: NEAR POINTS (Hybrid)
        # Geometry: mesh_to_sdf (Union) -> High Accuracy
        # Semantics: Heuristic (Per Class) -> Class determination
        # ---------------------------------------------------------------------
        # print("    Computing Near SDF (Hybrid)...")
        
        # A. Geometry (High Quality)
        near_sdf_union = compute_high_quality_sdf(near_points, union_mesh)
        
        # B. Semantics (Heuristic Check)
        # We need to filter "Which class is this point closest to?"
        # We can use the same Heuristic SDF logic, or just simple KDTree distance logic.
        # Simple KDTree check is enough for "closest class" if we trust Union SDF for sign.
        # But let's use Heuristic SDF to remain consistent with Vol strategy.
        
        near_sdf_heur_all = np.full((len(near_points), args.classes), 99.0, dtype=np.float32)
        for c in scaled_meshes.keys():
            verts, _ = scaled_meshes[c]
            # Use same heuristic for consistency
            near_sdf_heur_all[:, c-1] = compute_heuristic_sdf(
                near_points, verts, threshold=args.vol_threshold, batch_size=args.batch_size
            )
        
        # Heuristic Labels
        near_labels_prob = np.argmin(near_sdf_heur_all, axis=1) + 1
        
        # C. Fusion
        near_sdf = near_sdf_union
        near_labels = np.zeros(len(near_points), dtype=np.int8)
        
        # Only assign labels where Geometry says "Inside"
        is_inside_near = near_sdf < 0
        near_labels[is_inside_near] = near_labels_prob[is_inside_near]
        
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
                 surface_labels=surface_labels.astype(np.float32), # One-hot
                 vol_points=vol_points.astype(np.float32), 
                 vol_sdf=vol_sdf.astype(np.float32),
                 vol_labels=vol_labels, # Int (0-10)
                 near_points=near_points.astype(np.float32),
                 near_sdf=near_sdf.astype(np.float32),
                 near_labels=near_labels # Int (0-10)
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
    
    print(f"Processing {len(files)} files. Hybrid Final (Multi-Class).")
    print("="*60)
    
    for i, f in enumerate(files):
        print(f"[{i+1}/{len(files)}] {os.path.basename(f)}")
        process_file(f, args)
        if (i+1) % 10 == 0: gc.collect()
        
    print("Done.")

if __name__ == "__main__":
    main()
