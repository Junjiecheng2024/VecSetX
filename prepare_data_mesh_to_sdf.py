import os
import glob
import argparse
import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
import gc

# mesh_to_sdf 需要单独安装: pip install mesh-to-sdf
try:
    import mesh_to_sdf
    MESH_TO_SDF_AVAILABLE = True
except ImportError:
    MESH_TO_SDF_AVAILABLE = False
    print("Warning: mesh_to_sdf not available. Install with: pip install mesh-to-sdf")

def get_args():
    parser = argparse.ArgumentParser(description="Convert .nii.gz to .npz using mesh_to_sdf")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_surface_points", type=int, default=50000)
    parser.add_argument("--num_vol_points", type=int, default=50000)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    return parser.parse_args()

def compute_sdf_mesh_to_sdf(query_points, mesh):
    """
    Use mesh_to_sdf library for accurate SDF computation (CPU-only, no rendering)
    This works on headless servers without OpenGL/X11
    """
    if not MESH_TO_SDF_AVAILABLE:
        raise ImportError("mesh_to_sdf is required. Install with: pip install mesh-to-sdf")
    
    # Use 'depth' method instead of default 'scan' to avoid OpenGL rendering
    # This is slower but works on headless servers
    sdf_values = mesh_to_sdf.mesh_to_sdf(
        mesh, 
        query_points,
        surface_point_method='sample',  # Use sampling instead of scanning
        sign_method='depth',            # Use depth method (no OpenGL needed)
        scan_count=100,                 # Reasonable quality
        scan_resolution=400             # Reasonable resolution
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
        combined_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
        
        # Generate query points
        vol_points = np.random.uniform(-1, 1, (args.num_vol_points, 3)).astype(np.float32)
        
        # Compute SDF using mesh_to_sdf (THE PROFESSIONAL WAY!)
        print(f"    Computing vol_sdf with mesh_to_sdf...")
        vol_sdf = compute_sdf_mesh_to_sdf(vol_points, combined_mesh)
        
        # Near surface points
        near_points = surface_points + np.random.normal(0, 0.01, surface_points.shape).astype(np.float32)
        near_points = np.clip(near_points, -1, 1)
        
        print(f"    Computing near_sdf with mesh_to_sdf...")
        near_sdf = compute_sdf_mesh_to_sdf(near_points, combined_mesh)
        
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
        
        del combined_mesh, meshes, surface_points, vol_points, near_points
        del vol_sdf, near_sdf, all_verts, all_faces
        gc.collect()

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    args = get_args()
    
    if not MESH_TO_SDF_AVAILABLE:
        print("\n" + "="*70)
        print("ERROR: mesh_to_sdf library is not installed!")
        print("="*70)
        print("\nTo install:")
        print("  pip install mesh-to-sdf")
        print("\nOr with conda:")
        print("  conda install -c conda-forge pyembree")
        print("  pip install mesh-to-sdf")
        print("="*70)
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.nii.gz")))
    print(f"Found {len(files)} files.")
    
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(files)
    files = files[start:end]
    
    print(f"Processing {len(files)} files (index {start} to {end-1})")
    print(f"Settings: surface={args.num_surface_points}, vol={args.num_vol_points}")
    print(f"Using mesh_to_sdf for accurate SDF computation")
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
