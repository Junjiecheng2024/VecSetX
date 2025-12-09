import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import csv
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

def load_csv(csv_path):
    files = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 1:
                files.append(row[1])
    return files

def visualize_comparison(gt_slice, sample_id, slice_idx, axis, pc_proj=None, vol_proj=None, vol_lbl_proj=None):
    """
    Generates the comparison plot (GT + Points Overlay).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 1. GT Slice
    # Plot GT with a discrete colormap
    ax.imshow(gt_slice, cmap='nipy_spectral', vmin=0, vmax=10, interpolation='nearest')
    
    # GT Binary Count
    gt_bin = gt_slice > 0
    gt_count = np.sum(gt_bin)

    # Overlay Surface Points (White)
    if pc_proj is not None:
        res = gt_slice.shape[0]
        # Map [-1, 1] -> [0, res]
        r = (pc_proj[:, 0] + 1) * res / 2.0
        c = (pc_proj[:, 1] + 1) * res / 2.0
        
        valid = (r >= 0) & (r < res) & (c >= 0) & (c < res)
        if valid.any():
            ax.scatter(c[valid], r[valid], s=1, c='white', alpha=0.6, label='Surface')

    # Overlay Volume Points (Training Data)
    if vol_proj is not None and vol_lbl_proj is not None:
        res = gt_slice.shape[0]
        r = (vol_proj[:, 0] + 1) * res / 2.0
        c = (vol_proj[:, 1] + 1) * res / 2.0
        
        # Inside (negative SDF or label > 0) vs Outside (positive SDF or label 0)
        # Handle both SDF (float) and Labels (int)
        if vol_lbl_proj.dtype == np.int8 or vol_lbl_proj.dtype == np.int64 or vol_lbl_proj.dtype == int:
             is_inside = vol_lbl_proj > 0
        else:
             # SDF: Inside is Negative
             is_inside = vol_lbl_proj < 0
        
        valid = (r >= 0) & (r < res) & (c >= 0) & (c < res)
        
        # Plot Outside points (Blue)
        mask_out = valid & (~is_inside)
        if mask_out.any():
            ax.scatter(c[mask_out], r[mask_out], s=2, c='blue', alpha=0.3, label='Vol: Outside')
            
        # Plot Inside points (Magenta)
        mask_in = valid & is_inside
        if mask_in.any():
            ax.scatter(c[mask_in], r[mask_in], s=2, c='magenta', alpha=0.5, label='Vol: Inside')

    ax.set_title(f'Sample: {sample_id} | Axis {axis} | Slice {slice_idx}\nGT Voxels: {gt_count}')
    ax.axis('off')
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0, help='Index of sample to visualize')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to objaverse_val.csv or train.csv')
    parser.add_argument('--npz_dir', type=str, required=True, help='Directory with .npz files')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory with .nii.gz files')
    parser.add_argument('--output_dir', type=str, default='data_check_vis')
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--axis', type=int, default=2, help='Slice axis (0,1,2)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Get Filename
    files = load_csv(args.csv_path)
    if args.index >= len(files):
        print(f"Index {args.index} out of range (max {len(files)-1})")
        return
        
    filename = files[args.index]
    print(f"Checking Data Index {args.index}: {filename}")
    
    npz_path = os.path.join(args.npz_dir, filename + '.npz')
    gt_path = os.path.join(args.gt_dir, filename + '.nii.gz')
    
    # 2. Load NPZ
    if not os.path.exists(npz_path):
        print(f"NPZ not found: {npz_path}")
        return
        
    npz_data = np.load(npz_path)
    print("NPZ Keys:", list(npz_data.keys()))
    
    surf_pts = npz_data['surface_points'] # (N, 3)
    if 'surface_labels' in npz_data:
        surf_lbl = npz_data['surface_labels']
        
        # Determine Present Classes
        if surf_lbl.ndim == 2 and surf_lbl.shape[1] == 10:
            present_classes_idx = np.where(surf_lbl.sum(axis=0) > 0)[0]
            present_classes = present_classes_idx + 1 
        elif surf_lbl.ndim == 1:
            present_classes = np.unique(surf_lbl)
            present_classes = present_classes[present_classes > 0]
        else:
            present_classes = np.arange(1, 11)
    else:
        print("Warning: No surface_labels found. Assuming all classes present.")
        present_classes = np.arange(1, 11)

    print(f"Classes present in Surf: {present_classes}")
    
    # Load Volume Points
    vol_pts = None
    vol_data = None
    if 'vol_points' in npz_data:
        vol_pts = npz_data['vol_points']
        if 'vol_labels' in npz_data:
            vol_data = npz_data['vol_labels']
            print("Loaded vol_labels")
        elif 'vol_sdf' in npz_data:
            vol_data = npz_data['vol_sdf']
            print("Loaded vol_sdf. Range:", vol_data.min(), vol_data.max())
            
            # Simple stats
            pos = np.sum(vol_data > 0)
            neg = np.sum(vol_data < 0)
            print(f"Vol SDF Stats: Positive={pos} ({pos/len(vol_data)*100:.1f}%), Negative={neg} ({neg/len(vol_data)*100:.1f}%)")
        else:
            print("Warning: vol_points found but no labels/sdf!")

    # 3. Load GT and Recalculate Norm Params (Using ONLY present classes)
    if not os.path.exists(gt_path):
         print(f"GT NII not found: {gt_path}")
         return
         
    nii = nib.load(gt_path)
    img_data = nii.get_fdata()
    
    print("Normalization: Matching BBox...")
    mask = np.isin(img_data, present_classes)
    pts = np.argwhere(mask)
    
    if len(pts) == 0:
        print("Empty GT mask for present classes!")
        return
        
    shifts = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    centered_pts = pts - shifts
    max_dist = np.max(np.linalg.norm(centered_pts, axis=1))
    scale = 1.0 / max_dist if max_dist > 0 else 1.0
    
    print(f"Norm Params: Shift={shifts}, Scale={scale}")

    # 4. Create Evaluation Grid
    density = args.resolution
    x = np.linspace(-1, 1, density)
    y = np.linspace(-1, 1, density)
    z = np.linspace(-1, 1, density)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    grid_unit = np.stack([xv, yv, zv], axis=-1)
    
    # Map grid back to voxel space
    grid_voxel = (grid_unit / scale) + shifts
    coords_flat = grid_voxel.reshape(-1, 3).T
    
    print("Resampling GT slice...")
    gt_resampled_flat = map_coordinates(img_data, coords_flat, order=0, mode='constant', cval=0)
    gt_resampled = gt_resampled_flat.reshape(density, density, density)
    
    # 5. Slicing
    slice_thickness = 2.0 / density
    mid = density // 2
    
    # Prepare Projection Data
    if args.axis == 0:
        sl_gt = gt_resampled[mid, :, :]
        # Slice 0 (X) -> Show Y(1), Z(2)
        # On Slice check: X coord
        mask_surf = np.abs(surf_pts[:, 0]) < slice_thickness
        pc_proj = surf_pts[mask_surf][:, [1, 2]]
        
        if vol_pts is not None:
             mask_vol = np.abs(vol_pts[:, 0]) < slice_thickness
             vol_proj = vol_pts[mask_vol][:, [1, 2]]
             vol_lbl_proj = vol_data[mask_vol] if vol_data is not None else None
        else:
             vol_proj, vol_lbl_proj = None, None

    elif args.axis == 1:
        sl_gt = gt_resampled[:, mid, :]
        # Slice 1 (Y) -> Show X(0), Z(2)
        mask_surf = np.abs(surf_pts[:, 1]) < slice_thickness
        pc_proj = surf_pts[mask_surf][:, [0, 2]]
        
        if vol_pts is not None:
             mask_vol = np.abs(vol_pts[:, 1]) < slice_thickness
             vol_proj = vol_pts[mask_vol][:, [0, 2]]
             vol_lbl_proj = vol_data[mask_vol] if vol_data is not None else None
        else:
             vol_proj, vol_lbl_proj = None, None
             
    else: # axis 2
        sl_gt = gt_resampled[:, :, mid]
        # Slice 2 (Z) -> Show X(0), Y(1)
        mask_surf = np.abs(surf_pts[:, 2]) < slice_thickness
        pc_proj = surf_pts[mask_surf][:, [0, 1]]
        
        if vol_pts is not None:
             mask_vol = np.abs(vol_pts[:, 2]) < slice_thickness
             vol_proj = vol_pts[mask_vol][:, [0, 1]]
             vol_lbl_proj = vol_data[mask_vol] if vol_data is not None else None
        else:
             vol_proj, vol_lbl_proj = None, None

    # 6. Plot
    fig = visualize_comparison(sl_gt, filename, mid, args.axis, pc_proj=pc_proj, vol_proj=vol_proj, vol_lbl_proj=vol_lbl_proj)
    out_name = os.path.join(args.output_dir, f'{filename}_idx{args.index}_check.png')
    fig.savefig(out_name)
    print(f"Saved visualization to: {out_name}")

if __name__ == '__main__':
    main()
