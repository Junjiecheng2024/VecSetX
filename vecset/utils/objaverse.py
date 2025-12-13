
import os

import torch
from torch.utils import data

import numpy as np

import csv

import trimesh

# ============================================================================
# Class-Balanced Surface Sampling
# ============================================================================
def balanced_sample_surface(surface, surface_labels, target_size):
    """
    Sample surface points ensuring each class gets proportional representation.
    This helps the model learn small structures like Aorta, PA, PV better.
    
    Args:
        surface: (N, 3) surface points
        surface_labels: (N, C) one-hot or (N,) class indices
        target_size: number of points to sample
    
    Returns:
        indices: array of sampled indices
    """
    if surface_labels.ndim == 2:
        # One-hot: get class index for each point (0 = background if all zeros)
        class_idx = np.argmax(surface_labels, axis=1)
        # Check for background (all zeros in one-hot)
        is_background = surface_labels.sum(axis=1) == 0
        class_idx[is_background] = 0
    else:
        class_idx = surface_labels
    
    unique_classes = np.unique(class_idx)
    unique_classes = unique_classes[unique_classes > 0]  # Exclude background
    
    if len(unique_classes) == 0:
        # Fallback to random sampling if no valid classes
        return np.random.default_rng().choice(len(surface), target_size, replace=len(surface) < target_size)
    
    points_per_class = target_size // len(unique_classes)
    remainder = target_size % len(unique_classes)
    
    sampled_indices = []
    for i, cls in enumerate(unique_classes):
        cls_indices = np.where(class_idx == cls)[0]
        n_sample = points_per_class + (1 if i < remainder else 0)
        
        if len(cls_indices) >= n_sample:
            chosen = np.random.default_rng().choice(cls_indices, n_sample, replace=False)
        else:
            # If not enough points for this class, sample with replacement
            chosen = np.random.default_rng().choice(cls_indices, n_sample, replace=True)
        sampled_indices.extend(chosen)
    
    return np.array(sampled_indices)

class Objaverse(data.Dataset):
    def __init__(
        self, 
        split, 
        transform=None, 
        sdf_sampling=True, 
        sdf_size=4096, 
        surface_sampling=True, 
        surface_size=2048,
        dataset_folder='/ibex/project/c2281/objaverse',
        return_sdf=True,
        partial_prob=0.0,
        min_remove=0,
        max_remove=0,
        num_classes=10
        ):
        
        self.surface_size = surface_size

        self.transform = transform
        self.sdf_sampling = sdf_sampling
        self.sdf_size = sdf_size
        self.split = split
        self.partial_prob = partial_prob
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.num_classes = num_classes

        self.surface_sampling = surface_sampling

        self.npz_folder = dataset_folder
        self.normal_folder = dataset_folder.replace('objaverse', 'objaverse_normals')

        csv_path = os.path.join(os.path.dirname(__file__), 'objaverse_{}.csv'.format(split))
        print(f"Objaverse: Loading CSV from {csv_path}")
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            model_filenames = [(os.path.join(self.npz_folder, row[0], row[1]+'.npz'), row[2]) for row in reader]

        self.models = model_filenames
        
        self.return_sdf = return_sdf

    def __getitem__(self, idx):
        
        npz_path = self.models[idx][0]
        try:
            with np.load(npz_path) as data:
                vol_points = data['vol_points']
                vol_sdf = data['vol_sdf']
                near_points = data['near_points']
                near_sdf = data['near_sdf']
                surface = data['surface_points']  # Load surface points
                if 'surface_labels' in data:
                    surface_labels = data['surface_labels']
                    orig_cols = surface_labels.shape[1]
                    # Handle Padding if num_classes increased
                    if surface_labels.shape[1] < self.num_classes:
                        pad_size = self.num_classes - surface_labels.shape[1]
                        padding = np.zeros((surface_labels.shape[0], pad_size), dtype=surface_labels.dtype)
                        surface_labels = np.concatenate([surface_labels, padding], axis=1)
                else:
                    surface_labels = None
                    orig_cols = 0
                
                if 'vol_labels' in data:
                    vol_labels = data['vol_labels']
                else:
                    vol_labels = np.zeros(vol_points.shape[0], dtype=np.int8)

                if 'near_labels' in data:
                    near_labels = data['near_labels']
                else:
                    near_labels = np.zeros(near_points.shape[0], dtype=np.int8)
        except Exception as e:
            idx = np.random.randint(self.__len__())
            return self.__getitem__(idx)

        # Partial Input Masking Strategy (Random Class Dropout)
        # Apply BEFORE sampling to ensure we sample from remaining structure or just mask the sampled points?
        # Masking before sampling allows 'surface_size' to be filled with remaining points (if enough).
        # But if we mask after sampling, we might get fewer points. 
        # Better: Filter indices before sampling.
        
        valid_indices = None
        if self.split == 'train' and self.partial_prob > 0:
             # Phase 3: Structural Subsets
             # If min_remove/max_remove are set, use random dropout (Phase 2 logic)
             # If min_remove=0, check for subset mode logic (Phase 3)
             
             # Defined Subsets (1-based class IDs)
             # 1:Myo, 2:LA, 3:LV, 4:RA, 5:RV, 6:Ao, 7:PA, 8:LAA, 9:Cor, 10:PV
             SUBSETS = {
                 'left_heart': [1, 2, 3, 6, 8, 10], 
                 'right_heart': [4, 5, 7],
                 'four_chambers': [2, 3, 4, 5],
                 'great_vessels': [6, 7, 9]
             }

             if self.max_remove > 0:
                 # Phase 2 Logic: Random Dropout
                 if np.random.rand() < self.partial_prob:
                     if surface_labels is not None:
                         if surface_labels.ndim == 2:
                             present_classes = np.where(surface_labels.sum(axis=0) > 0)[0] 
                             if len(present_classes) > self.min_remove:
                                 n_remove = np.random.randint(self.min_remove, min(self.max_remove + 1, len(present_classes)))
                                 remove_cols = np.random.choice(present_classes, n_remove, replace=False)
                                 mask_remove = surface_labels[:, remove_cols].sum(axis=1) > 0
                                 valid_indices = np.where(~mask_remove)[0]
                         elif surface_labels.ndim == 1:
                             present_classes = np.unique(surface_labels)
                             present_classes = present_classes[present_classes > 0]
                             if len(present_classes) > self.min_remove:
                                 n_remove = np.random.randint(self.min_remove, min(self.max_remove + 1, len(present_classes)))
                                 remove_classes = np.random.choice(present_classes, n_remove, replace=False)
                                 mask_remove = np.isin(surface_labels, remove_classes)
                                 valid_indices = np.where(~mask_remove)[0]
             
             else:
                 # Phase 3 Logic: Structural Subset (Co-occurrence)
                 # If no max_remove, but partial_prob > 0, we assume Subset Mode
                 # We randomly pick ONE subset to KEEP.
                 if np.random.rand() < self.partial_prob:
                     # Randomly select a subset definition
                     subset_name = np.random.choice(list(SUBSETS.keys()))
                     keep_classes = SUBSETS[subset_name]
                     
                     if surface_labels is not None:
                         if surface_labels.ndim == 2:
                             # Keep points if they belong to ANY of the keep_classes
                             # keep_classes are 1-based. indices are 0-based.
                             keep_cols = [c-1 for c in keep_classes if c-1 < self.num_classes] # Updated < 10 to < num_classes check
                             if len(keep_cols) > 0:
                                 mask_keep = surface_labels[:, keep_cols].sum(axis=1) > 0
                                 valid_indices = np.where(mask_keep)[0]
                                 
                         elif surface_labels.ndim == 1:
                             mask_keep = np.isin(surface_labels, keep_classes)
                             valid_indices = np.where(mask_keep)[0]

        if valid_indices is not None and len(valid_indices) < 50:
             # Safety fallback: if we removed almost everything, revert or keep at least something
             valid_indices = None
             
        if self.surface_sampling:
            if valid_indices is not None:
                # Sample from valid subset
                if len(valid_indices) >= self.surface_size:
                    ind = np.random.default_rng().choice(valid_indices, self.surface_size, replace=False)
                else:
                    ind = np.random.default_rng().choice(valid_indices, self.surface_size, replace=True)
            elif self.split == 'train' and surface_labels is not None:
                # ================================================================
                # Class-Balanced Sampling for Training
                # ================================================================
                ind = balanced_sample_surface(surface, surface_labels, self.surface_size)
            else:
                # Standard random sampling for validation
                ind = np.random.default_rng().choice(surface.shape[0], self.surface_size, replace=False)
                
            surface = surface[ind]
            if surface_labels is not None:
                surface_labels = surface_labels[ind]
            # surface_normals = surface_normals[ind]
            # surface_normals = trimesh.unitize(surface_normals)
        
        surface = torch.from_numpy(surface)
        if surface_labels is not None:
            surface_labels = torch.from_numpy(surface_labels)
            surface = torch.cat([surface, surface_labels], dim=-1)

        # ... (rest of the file remains same)

        if self.sdf_sampling:
            ### make sure balanced sampling
            # Training code expects 1024 vol + 1024 near = 2048 total
            # So we sample sdf_size//4 for each (4096//4 = 1024)
            pos_vol_id = (vol_sdf<0).reshape(-1)

            # Sample positive (inside) points
            if pos_vol_id.sum() > 0:
                replace = pos_vol_id.sum() < self.sdf_size//4
                pos_ind = np.random.default_rng().choice(pos_vol_id.sum(), self.sdf_size//4, replace=replace)
                pos_vol_points = vol_points[pos_vol_id][pos_ind]
                pos_vol_sdf = vol_sdf[pos_vol_id][pos_ind]
                pos_vol_labels = vol_labels[pos_vol_id][pos_ind]
            else:
                # Fallback to random sampling
                pos_ind = np.random.default_rng().choice(vol_points.shape[0], self.sdf_size//4, replace=True)
                pos_vol_points = vol_points[pos_ind]
                pos_vol_sdf = vol_sdf[pos_ind]
                pos_vol_labels = vol_labels[pos_ind]

            # Sample negative (outside) points
            neg_vol_id = (vol_sdf>=0).reshape(-1)
            
            if neg_vol_id.sum() > 0:
                replace = neg_vol_id.sum() < self.sdf_size//4
                neg_ind = np.random.default_rng().choice(neg_vol_id.sum(), self.sdf_size//4, replace=replace)
                neg_vol_points = vol_points[neg_vol_id][neg_ind]
                neg_vol_sdf = vol_sdf[neg_vol_id][neg_ind]
                neg_vol_labels = vol_labels[neg_vol_id][neg_ind]
            else:
                # Fallback
                neg_ind = np.random.default_rng().choice(vol_points.shape[0], self.sdf_size//4, replace=True)
                neg_vol_points = vol_points[neg_ind]
                neg_vol_sdf = vol_sdf[neg_ind]
                neg_vol_labels = vol_labels[neg_ind]
            
            # Concatenate pos and neg samples
            vol_points_sampled = np.concatenate([pos_vol_points, neg_vol_points], axis=0)
            vol_sdf_sampled = np.concatenate([pos_vol_sdf, neg_vol_sdf], axis=0)
            vol_labels_sampled = np.concatenate([pos_vol_labels, neg_vol_labels], axis=0)
            
            # Sample fixed number of near points to ensure consistent batch sizes
            # near_points usually has ~50k points, we need exactly sdf_size//2 (1024)
            n_near_points = near_points.shape[0]
            if n_near_points >= self.sdf_size//4:
                near_indices = np.random.default_rng().choice(n_near_points, self.sdf_size//4, replace=False)
            else:
                # If not enough, sample with replacement
                near_indices = np.random.default_rng().choice(n_near_points, self.sdf_size//4, replace=True)
            
            near_points_sampled = near_points[near_indices]
            near_sdf_sampled = near_sdf[near_indices]
            near_labels_sampled = near_labels[near_indices]
            
            # Convert to torch and ensure correct shapes
            vol_points_sampled = torch.from_numpy(vol_points_sampled).float()
            vol_sdf_sampled = torch.from_numpy(vol_sdf_sampled).float().flatten()  # Ensure 1D
            vol_labels_sampled = torch.from_numpy(vol_labels_sampled).long().flatten()  # Ensure 1D
            
            near_points_sampled = torch.from_numpy(near_points_sampled).float()
            near_sdf_sampled = torch.from_numpy(near_sdf_sampled).float().flatten()  # Ensure 1D
            near_labels_sampled = torch.from_numpy(near_labels_sampled).long().flatten()  # Ensure 1D

            # Concatenate vol and near (now both are sdf_size//2 = 1024 points each)
            points = torch.cat([vol_points_sampled, near_points_sampled], dim=0)
            
            # Stack SDF and Labels: (N, 2)
            vol_target = torch.stack([vol_sdf_sampled, vol_labels_sampled.float()], dim=1)
            near_target = torch.stack([near_sdf_sampled, near_labels_sampled.float()], dim=1)
            
            sdf = torch.cat([vol_target, near_target], dim=0)

        if self.transform:
            # Note: transform usually expects (surface, points). 
            # If surface has labels, transform might fail if it expects 3 coords.
            # We should split, transform, and concat back if transform modifies coords.
            # But Objaverse doesn't seem to have complex transforms by default in main_ae.py
            # Let's assume transform handles it or is None.
            # In main_ae.py: dataset_train = Objaverse(..., transform=None, ...)
            # So it is None.
            surface, points = self.transform(surface, points)

        ## random rotation
        if self.split == 'train':
            # Split labels if present
            if surface.shape[-1] > 3:
                surface_coords = surface[:, :3]
                surface_labels_tensor = surface[:, 3:]
            else:
                surface_coords = surface
                surface_labels_tensor = None

            perm = torch.randperm(3)
            points = points[:, perm]
            surface_coords = surface_coords[:, perm]

            negative = torch.randint(2, size=(3,)) * 2 - 1
            points *= negative[None]
            surface_coords *= negative[None]

            roll = torch.randn(1)
            yaw = torch.randn(1)
            pitch = torch.randn(1)

            tensor_0 = torch.zeros(1)
            tensor_1 = torch.ones(1)

            RX = torch.stack([
                            torch.stack([tensor_1, tensor_0, tensor_0]),
                            torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                            torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)

            RY = torch.stack([
                            torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                            torch.stack([tensor_0, tensor_1, tensor_0]),
                            torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3,3)

            RZ = torch.stack([
                            torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                            torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                            torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)

            R = torch.mm(RZ, RY)
            R = torch.mm(R, RX)

            points = torch.mm(points, R).detach()
            surface_coords = torch.mm(surface_coords, R).detach()
            # surface_normals = torch.mm(surface_normals, R).detach()

            # Re-concatenate
            if surface_labels_tensor is not None:
                surface = torch.cat([surface_coords, surface_labels_tensor], dim=-1)
            else:
                surface = surface_coords
              
        if self.return_sdf is False: # return occupancies (sign) instead
            sdf = (sdf<0).float()
        
        # Create Class Mask (for Phase 3 Label Conflict Mitigation)
        # mask shape: (num_classes,)
        valid_class_mask = torch.ones(self.num_classes, dtype=torch.float32)
        
        # If we loaded a file with fewer columns than num_classes, mask out the extra columns
        if 'orig_cols' in locals() and orig_cols > 0 and orig_cols < self.num_classes:
            valid_class_mask[orig_cols:] = 0.0
        
        return points, sdf, surface, self.models[idx][1], npz_path, valid_class_mask#, surface_normals

    def __len__(self):
        return len(self.models)
