import os
import glob
import argparse
import numpy as np
import nibabel as nib
import trimesh
import torch
from skimage import measure
from scipy.spatial import cKDTree

def get_args():
    parser = argparse.ArgumentParser(description="Convert .nii.gz masks to .npz for VecSetX")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .nii.gz files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npz files")
    parser.add_argument("--num_surface_points", type=int, default=100000, help="Total number of surface points to sample")
    parser.add_argument("--num_vol_points", type=int, default=100000, help="Total number of volume points to sample")
    parser.add_argument("--classes", type=int, default=10, help="Number of classes")
    return parser.parse_args()

def process_file(file_path, args):
    try:
        # Load NIfTI file
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Normalize spacing if needed (assuming isotropic for now or handling via affine)
        # For simplicity in this script, we work in voxel coordinates and then apply affine if needed.
        # However, VecSetX usually expects normalized coordinates [-1, 1].
        
        # 1. Extract Surface Points and Labels
        surface_points_list = []
        surface_labels_list = []
        
        # We need to handle each class (1 to 10)
        # Assuming labels are 1-based indices in the mask
        
        total_surface_area = 0
        meshes = {}
        
        for c in range(1, args.classes + 1):
            mask = (data == c)
            if np.sum(mask) == 0:
                continue
                
            # Marching Cubes to get mesh
            try:
                verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
                # Apply affine to get real world coordinates
                # verts = nib.affines.apply_affine(img.affine, verts) 
                # For now, let's stick to voxel coordinates and normalize later to keep it simple and robust
                
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                meshes[c] = mesh
                total_surface_area += mesh.area
            except Exception as e:
                print(f"Warning: Failed to extract mesh for class {c} in {file_path}: {e}")
                continue

        if total_surface_area == 0:
            print(f"Error: No surface found in {file_path}")
            return

        # Sample points proportional to area
        for c, mesh in meshes.items():
            n_samples = int(args.num_surface_points * (mesh.area / total_surface_area))
            if n_samples == 0: continue
            
            points, _ = trimesh.sample.sample_surface(mesh, n_samples)
            surface_points_list.append(points)
            
            # Create one-hot label
            label = np.zeros((n_samples, args.classes), dtype=np.float32)
            label[:, c-1] = 1.0 # 0-indexed for one-hot
            surface_labels_list.append(label)
            
        if not surface_points_list:
            return

        surface_points = np.concatenate(surface_points_list, axis=0)
        surface_labels = np.concatenate(surface_labels_list, axis=0)
        
        # Normalize surface points to [-1, 1]
        # Center and scale
        min_bound = surface_points.min(axis=0)
        max_bound = surface_points.max(axis=0)
        center = (min_bound + max_bound) / 2
        scale = (max_bound - min_bound).max()
        
        surface_points = (surface_points - center) / (scale / 2) # Scale to [-1, 1] range roughly
        # Actually VecSetX normalization: 
        # shifts = (surface.max(axis=0) + surface.min(axis=0)) / 2
        # surface = surface - shifts
        # distances = np.linalg.norm(surface, axis=1)
        # scale = 1 / np.max(distances)
        # surface *= scale
        
        # Let's use the logic from infer.py to be consistent
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
        # We need points in the volume [-1, 1]
        
        vol_points = np.random.uniform(-1, 1, (args.num_vol_points, 3)).astype(np.float32)
        
        # Combine all meshes for SDF calculation
        combined_verts = []
        combined_faces = []
        combined_normals = []
        vertex_offset = 0
        
        for c, mesh in meshes.items():
            v = mesh.vertices
            # Normalize vertices same as points
            v = v - shifts
            v *= scale_factor
            
            combined_verts.append(v)
            combined_faces.append(mesh.faces + vertex_offset)
            # We need vertex normals for sign approximation
            combined_normals.append(mesh.vertex_normals)
            vertex_offset += len(v)
            
        all_verts = np.concatenate(combined_verts)
        all_normals = np.concatenate(combined_normals)
        
        # Build KDTree on vertices (faster than surface points if mesh is dense enough, or use surface points if we have normals for them)
        # Marching cubes vertices are good.
        tree = cKDTree(all_verts)
        
        def compute_approx_sdf(query_points):
            dists, idxs = tree.query(query_points)
            nearest_normals = all_normals[idxs]
            # Vector from surface to query
            vec = query_points - all_verts[idxs]
            # Sign: dot product
            # If dot > 0, outside (assuming normals point out). 
            signs = np.sign(np.sum(vec * nearest_normals, axis=1))
            # Fix sign for points exactly on surface (dist=0)
            signs[dists < 1e-6] = 1
            return dists * signs

        # For volume points
        vol_sdf = compute_approx_sdf(vol_points)
        
        # Sample near surface
        near_points = surface_points + np.random.normal(0, 0.01, surface_points.shape)
        near_sdf = compute_approx_sdf(near_points)
        
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
        
        print(f"Saved {save_path}")

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(args.input_dir, "*.nii.gz"))
    print(f"Found {len(files)} files.")
    
    for f in files:
        process_file(f, args)

if __name__ == "__main__":
    main()
