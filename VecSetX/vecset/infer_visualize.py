
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import csv
import nibabel as nib
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator

# Robust imports
try:
    from vecset.models import autoencoder
except ImportError:
    try:
        from models import autoencoder
    except ImportError:
        print("Cannot import models. Please set PYTHONPATH.")
        sys.exit(1)

def load_csv(csv_path):
    files = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Row match Objaverse: ['', filename, ...]
            if len(row) > 1:
                files.append(row[1])
    return files

def visualize_comparison(gt_slice, pred_slice, sample_id, slice_idx, axis, pc_proj=None):
    """Generates the comparison plot."""
    # GT Slice: (H, W), entries 0-10
    # Pred Slice: (H, W), entries 0-10
    
    # Create Difference Map (Binary mismatch for now, or Class mismatch)
    # Let's do simple binary Inside/Outside mismatch first
    gt_bin = gt_slice > 0
    pred_bin = pred_slice > 0
    
    diff = np.zeros_like(gt_slice, dtype=np.int8)
    diff[gt_bin & ~pred_bin] = -1 # False Negative (Red)
    diff[~gt_bin & pred_bin] = 1  # False Positive (Green)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. GT
    axes[0].imshow(gt_slice, cmap='nipy_spectral', vmin=0, vmax=10, interpolation='nearest')
    
    # Overlay PC Points if available
    if pc_proj is not None:
        # pc_proj is (N, 2) in [-1, 1]. Need to map to pixel indices [0, Resolution].
        # Resolution is gt_slice.shape[0]
        res = gt_slice.shape[0]
        # x_idx = (x + 1)/2 * res
        # y_idx = (y + 1)/2 * res
        # Note: imshow axis 0 is Y (rows), axis 1 is X (cols).
        # We need to check projection axes carefully.
        # Assuming pc_proj is [row_coord, col_coord] in range [-1, 1]
        
        # Map [-1, 1] -> [0, res]
        r = (pc_proj[:, 0] + 1) * res / 2.0
        c = (pc_proj[:, 1] + 1) * res / 2.0
        
        # Clip
        valid = (r >= 0) & (r < res) & (c >= 0) & (c < res)
        axes[0].scatter(c[valid], r[valid], s=1, c='white', alpha=0.5, label='Input PC')
        # axes[0].legend()

    axes[0].set_title(f'GT Mask + PC Overlay\n(Val > 0: {np.sum(gt_bin)})')
    axes[0].axis('off')
    
    # 2. Pred
    axes[1].imshow(pred_slice, cmap='nipy_spectral', vmin=0, vmax=10, interpolation='nearest')
    axes[1].set_title(f'Pred Mask\n(Val > 0: {np.sum(pred_bin)})')
    axes[1].axis('off')
    
    # 3. Binary Overlay
    # Green = GT, Red = Pred, Yellow = Overlap
    # Wait, Reference script: Green=Orig, Red=Refined.
    # Let's stick to standard: Green=GT, Red=Pred.
    overlay = np.zeros((*gt_slice.shape, 3))
    overlay[gt_bin] = [0, 1, 0] # GT Green
    overlay[pred_bin] = [1, 0, 0] # Pred Red
    overlay[gt_bin & pred_bin] = [1, 1, 0] # Overlap Yellow
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Binary)\nGreen=GT, Red=Pred, Yellow=Match')
    axes[2].axis('off')
    
    # 4. Difference
    # Red = Missing (FN), Green = Added (FP)
    from matplotlib.colors import ListedColormap
    cmap_diff = ListedColormap(['red', 'black', 'green'])
    axes[3].imshow(diff, cmap=cmap_diff, vmin=-1, vmax=1, interpolation='nearest')
    axes[3].set_title('Difference\nRed=Missing, Green=Extraneous')
    axes[3].axis('off')
    
    plt.suptitle(f'Sample: {sample_id} | Axis {axis} | Slice {slice_idx}', fontsize=16)
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0, help='Index of sample to visualize')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to objaverse_val.csv or train.csv')
    parser.add_argument('--npz_dir', type=str, required=True, help='Directory with .npz files')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory with .nii.gz files')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='vis_output')
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--axis', type=int, default=2, help='Slice axis (0,1,2)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='learnable_vec1024x16_dim1024_depth24_nb')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Get Filename
    files = load_csv(args.csv_path)
    if args.index >= len(files):
        print(f"Index {args.index} out of range (max {len(files)-1})")
        return
        
    filename = files[args.index]
    print(f"Visualizing Index {args.index}: {filename}")
    
    npz_path = os.path.join(args.npz_dir, filename + '.npz')
    gt_path = os.path.join(args.gt_dir, filename + '.nii.gz')
    
    # Load NPZ first to get present classes
    if not os.path.exists(npz_path):
        print(f"NPZ not found: {npz_path}")
        return
        
    npz_data = np.load(npz_path)
    surf_pts = npz_data['surface_points'] # (N, 3)
    surf_lbl = npz_data['surface_labels'] # (N, 10 or N, 1)
    
    # Handle Label Shape
    if surf_lbl.ndim == 2 and surf_lbl.shape[1] == 10:
        # One-hot
        present_classes_idx = np.where(surf_lbl.sum(axis=0) > 0)[0]
        present_classes = present_classes_idx + 1 # 1-based class IDs
    elif surf_lbl.ndim == 1:
        # Integer labels
        present_classes = np.unique(surf_lbl)
        present_classes = present_classes[present_classes > 0]
    else:
        # Fallback
        present_classes = np.arange(1, 11)
        
    print(f"Classes present in NPZ: {present_classes}")
    
    # Concatenate for model input
    # Ensure surf_lbl is (N, 10) for concatenation if model expects 13 channels
    if surf_lbl.ndim == 1:
        # Convert to one-hot if it was int
        n_pts = len(surf_lbl)
        one_hot = np.zeros((n_pts, 10), dtype=np.float32)
        # valid indices
        valid = (surf_lbl > 0) & (surf_lbl <= 10)
        if np.any(valid):
            # indices -1 because classes are 1..10, cols 0..9
            cols = surf_lbl[valid].astype(int) - 1
            one_hot[valid, cols] = 1.0
        surf_lbl = one_hot
        
    surf_pts_input = np.concatenate([surf_pts, surf_lbl], axis=1)

    # 3. Load GT and Recalculate Norm Params (Using ONLY present classes)
    import skimage.measure
    if not os.path.exists(gt_path):
         print(f"GT NII not found: {gt_path}")
         return
         
    nii = nib.load(gt_path)
    img_data = nii.get_fdata()
    
    print("Re-calculating normalization parameters from GT (Matched Classes)...")
    # Filter for present classes
    mask = np.isin(img_data, present_classes)
    pts = np.argwhere(mask) 
    
    if len(pts) == 0:
        print("Empty GT mask (after filtering)!")
        return

    # Convert to physical coords? prepare_data used `mesh.vertices` which are usually in voxel/affine coordinates.
    # Wait, `skimage.measure.marching_cubes(class_mask, level=0.5)` returns verts in index space.
    # So `mesh.vertices` are roughly voxel indices.
    # So `pts` (indices) is a good approximation of the geometry.
    # Let's use `pts` min/max to emulate `mesh.bounds`.
    
    # Correction: Mesh verts are continuous. Indices are discrete.
    # `shifts` = (max + min) / 2
    # `max_dist` = max(norm(pts - shifts))
    
    if len(pts) == 0:
        print("Empty GT mask!")
        return
        
    shifts = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    centered_pts = pts - shifts
    max_dist = np.max(np.linalg.norm(centered_pts, axis=1))
    scale = 1.0 / max_dist if max_dist > 0 else 1.0
    
    print(f"Norm Params: Shift={shifts}, Scale={scale}")
    
    # 4. Create Evaluation Grid in Unit Cube
    density = args.resolution
    x = np.linspace(-1, 1, density)
    y = np.linspace(-1, 1, density)
    z = np.linspace(-1, 1, density)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    grid_unit = np.stack([xv, yv, zv], axis=-1) # (D, D, D, 3)
    
    # 5. Map Grid back to Voxel Space to sample GT
    # grid_unit = (grid_voxel - shifts) * scale
    # grid_voxel = (grid_unit / scale) + shifts
    grid_voxel = (grid_unit / scale) + shifts
    
    # Sample GT at grid_voxel using Nearest Neighbor
    # img_data is (X, Y, Z).
    # We need to interpolation.
    # RegularGridInterpolator expects (x, y, z).
    
    # Optimize: Use `map_coordinates` from scipy
    from scipy.ndimage import map_coordinates
    # map_coordinates wants coordinates as (3, N)
    coords_flat = grid_voxel.reshape(-1, 3).T # (3, N_grid)
    
    print("Sampling GT onto visualization grid...")
    # order=0 is nearest neighbor
    gt_resampled_flat = map_coordinates(img_data, coords_flat, order=0, mode='constant', cval=0)
    gt_resampled = gt_resampled_flat.reshape(density, density, density)
    
    # Model Loading (Moved after NPZ check)
    model = autoencoder.__dict__[args.model_name](pc_size=8192, input_dim=13) 
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model' in state: state = state['model']
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # 4. Create Evaluation Grid in Unit Cube
    
    # Chunking
    block = 100000
    all_sdf = []
    all_logits = []
    # Prepare tensors for inference
    if len(surf_pts_input) > 8192:
        idx = np.random.choice(len(surf_pts_input), 8192, replace=False)
        surf_input = surf_pts_input[idx]
    else:
        surf_input = surf_pts_input

    surf_tensor = torch.from_numpy(surf_input).unsqueeze(0).to(device)
    grid_tensor = torch.from_numpy(grid_unit.reshape(-1, 3).astype(np.float32)).unsqueeze(0).to(device)
    
    # For visualization overlay
    surf_pts_vis = surf_pts_input[:, :3]

    with torch.no_grad():
        bottleneck = model.encode(surf_tensor)
        latent = model.learn(bottleneck['x'])
        
        N = grid_tensor.shape[1]
        for i in range(0, N, block):
            chunk = grid_tensor[:, i:i+block, :]
            out = model.decode(latent, chunk) # (1, M, 11)
            
            all_sdf.append(out[:, :, 0].cpu().numpy())
            all_logits.append(out[:, :, 1:].cpu().numpy())
            
    # Assemble
    pred_sdf = np.concatenate(all_sdf, axis=1).reshape(density, density, density)
    pred_logits = np.concatenate(all_logits, axis=1).reshape(density, density, density, 10)
    
    # Pred Mask Logic
    # Class = argmax(logits) + 1
    # But only if SDF < 0
    pred_class = np.argmax(pred_logits, axis=-1) + 1
    pred_mask = np.zeros_like(pred_class, dtype=np.uint8)
    
    inside_mask = pred_sdf < 0
    pred_mask[inside_mask] = pred_class[inside_mask]
    
    # 7. Slice and Plot
    slice_thickness = 2.0 / density
    mid = density // 2
    if args.axis == 0:
        sl_gt = gt_resampled[mid, :, :]
        sl_pred = pred_mask[mid, :, :]
        # Project PC: x=0 (slice axis). Show (y, z) -> (row, col)
        # Verify meshgrid order: indexing='ij' -> x, y, z
        # Slice 0 (X) -> show Y, Z?
        # Y corresponds to Axis 1. Z corresponds to Axis 2.
        # But imshow usually expects (Row, Col). 
        # Let's standardise: Row=Axis Y, Col=Axis Z. (Or vice versa)
        # slice coords usually: [mid, :, :] keeps dims 1 and 2.
        # So Row=Dim 1 (Y), Col=Dim 2 (Z).
        
        # Filter PC points near specific slice
        # Points in [-1, 1]. Mid slice is at 0.
        # Thickness of slice? 2/density.
        slice_thickness = 2.0 / density
        on_slice = np.abs(surf_pts_vis[:, 0]) < slice_thickness
        pc_proj = surf_pts_vis[on_slice][:, [1, 2]] # Y, Z
        
    elif args.axis == 1:
        sl_gt = gt_resampled[:, mid, :]
        sl_pred = pred_mask[:, mid, :]
        # Slice 1 (Y) -> show X (Row), Z (Col)
        on_slice = np.abs(surf_pts_vis[:, 1]) < slice_thickness
        pc_proj = surf_pts_vis[on_slice][:, [0, 2]] # X, Z
        
    else: # axis 2
        sl_gt = gt_resampled[:, :, mid]
        sl_pred = pred_mask[:, :, mid]
        # Slice 2 (Z) -> show X (Row), Y (Col)
        on_slice = np.abs(surf_pts_vis[:, 2]) < slice_thickness
        pc_proj = surf_pts_vis[on_slice][:, [0, 1]] # X, Y
        
    fig = visualize_comparison(sl_gt, sl_pred, filename, mid, args.axis, pc_proj=pc_proj)
    out_name = os.path.join(args.output_dir, f'{filename}_idx{args.index}_vis.png')
    fig.savefig(out_name)
    print(f"Saved: {out_name}")

if __name__ == '__main__':
    main()
