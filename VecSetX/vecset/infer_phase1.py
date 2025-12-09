
import argparse
import torch
import numpy as np
import mcubes
import trimesh
import os
import sys
import os

# Add current directory to path if needed or handle package
try:
    from vecset.models import autoencoder
except ImportError:
    try:
        from models import autoencoder
    except ImportError:
        print("Cannot import models. Please set PYTHONPATH to include VecSetX directory.")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True, help='Path to input .npz file (must contain surface_points)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--output', type=str, default='recon.ply', help='Output mesh path')
    parser.add_argument('--resolution', type=int, default=128, help='Grid resolution for MC')
    parser.add_argument('--model_name', type=str, default='learnable_vec1024x16_dim1024_depth24_nb')
    parser.add_argument('--input_dim', type=int, default=13)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load Data
    print(f"Loading data from {args.npz_path}...")
    data = np.load(args.npz_path)
    # Surface points: (N, 13)
    surf_pts = data['surface_points']
    surf_lbl = data['surface_labels']
    surface_points = np.concatenate([surf_pts, surf_lbl], axis=1)
    
    # Random sample 8192 if needed, or take all
    if surface_points.shape[0] > 8192:
        idx = np.random.choice(surface_points.shape[0], 8192, replace=False)
        surface_points = surface_points[idx]
    
    # Prepare batch (B=1)
    surface_tensor = torch.from_numpy(surface_points).unsqueeze(0).to(device) # (1, N, 13)

    # 2. Load Model
    print(f"Loading model {args.model_name}...")
    # NOTE: We must match the training configuration exactly
    model = autoencoder.__dict__[args.model_name](pc_size=8192, input_dim=args.input_dim)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # Handle 'model' key or direct state_dict
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 3. Create Grid
    print(f"Generating SDF grid ({args.resolution}^3)...")
    density = args.resolution
    gap = 2.0 / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    
    # (3, D*D*D)
    grid_coords = np.stack([xv, yv, zv]).reshape(3, -1).T 
    grid_tensor = torch.from_numpy(grid_coords.astype(np.float32)).unsqueeze(0).to(device) # (1, M, 3)

    # 4. Inference (Batched Grid)
    block_size = 100000 # Query in chunks to save memory
    all_sdf = []
    
    with torch.no_grad():
        # Encode once
        # surface_tensor is (1, 8192, 13)
        bottleneck = model.encode(surface_tensor)
        latent = model.learn(bottleneck['x'])
        
        # Decode grid
        N = grid_tensor.shape[1]
        for i in range(0, N, block_size):
            chunk = grid_tensor[:, i:i+block_size, :]
            # decoder expects (B, M, 3)
            # output is (B, M, 11) -> We need channel 0 (SDF)
            # outputs = model.decode(latent, chunk) <-- This returns (B, M, 11)
            # Wait, verify autoencoder.py: decode -> to_outputs which is Linear(dim, 11)
            
            out_chunk = model.decode(latent, chunk)
            sdf_chunk = out_chunk[:, :, 0] # (1, M)
            all_sdf.append(sdf_chunk.cpu().numpy())
            
    sdf_volume = np.concatenate(all_sdf, axis=1).reshape(density+1, density+1, density+1)

    # 5. Marching Cubes
    print("Running Marching Cubes...")
    # Note: MC usually looks for level set 0. 
    # Our data: inside < 0, outside > 0.
    
    # Important: Check if we need to flip sign?
    # Usually MC assumes inside is positive? No, MC finds the 0 crossing.
    
    # mcubes.marching_cubes(volume, level)
    verts, faces = mcubes.marching_cubes(sdf_volume, 0.0)
    
    # Transform verts back to [-1, 1]
    # step size in index space is 1
    # -1 corresponds to index 0
    # 1 corresponds to index density
    # coordinate = -1 + index * gap
    
    verts = -1.0 + verts * gap
    
    # 6. Save
    print(f"Saving mesh to {args.output}...")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(args.output)
    print("Done.")

if __name__ == '__main__':
    main()
