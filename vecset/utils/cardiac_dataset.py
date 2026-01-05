"""
Per-Class Cardiac Dataset Loader

This DataLoader is designed for the per-class binary implicit approach:
- Each class is treated as an independent binary implicit task
- No one-hot encoding - class info is implicit from data grouping
- Supports the directory structure from prepare_data_v2.py

Data structure expected:
    dataset_dir/
        case_001/
            class_1/
                surface_pts.npy   (N, 3)
                near_pts.npy      (N, 3)
                near_sdf.npy      (N, 1)
                vol_pts.npy       (N, 3)
                vol_sdf.npy       (N, 1)
                stats.json
            class_2/
                ...
            metadata.json
        case_002/
            ...
"""

import os
import json
import torch
from torch.utils import data
import numpy as np


class CardiacPerClass(data.Dataset):
    """
    Dataset for per-class binary implicit learning.
    
    Each sample is a (case, class) pair, allowing the model to learn
    each anatomical structure as an independent binary SDF task.
    """
    
    def __init__(
        self,
        dataset_folder,
        split='train',
        classes=None,  # List of class IDs to include, e.g. [1] for Myocardium only
        surface_size=8192,
        sdf_size=4096,
        transform=None,
        augment=True,
        return_sdf=True,
        val_ratio=0.1,
        seed=42,
    ):
        """
        Args:
            dataset_folder: Path to the dataset directory
            split: 'train' or 'val'
            classes: List of class IDs to include (e.g., [1] for Myocardium only)
            surface_size: Number of surface points to sample
            sdf_size: Number of SDF query points to sample
            transform: Optional transform to apply
            augment: Whether to apply random rotation augmentation
            return_sdf: If True, return SDF values; if False, return occupancy
            val_ratio: Ratio of data to use for validation
            seed: Random seed for train/val split
        """
        super().__init__()
        
        self.dataset_folder = dataset_folder
        self.split = split
        self.surface_size = surface_size
        self.sdf_size = sdf_size
        self.transform = transform
        self.augment = augment and (split == 'train')
        self.return_sdf = return_sdf
        
        # Default to class 1 (Myocardium) if not specified
        self.classes = classes if classes is not None else [1]
        
        # Find all valid (case, class) pairs
        self.samples = []
        case_dirs = sorted([d for d in os.listdir(dataset_folder) 
                           if os.path.isdir(os.path.join(dataset_folder, d))])
        
        for case_dir in case_dirs:
            case_path = os.path.join(dataset_folder, case_dir)
            
            for cls_id in self.classes:
                class_dir = os.path.join(case_path, f'class_{cls_id}')
                if os.path.isdir(class_dir):
                    # Check all required files exist
                    required_files = ['surface_pts.npy', 'near_pts.npy', 'near_sdf.npy',
                                     'vol_pts.npy', 'vol_sdf.npy']
                    if all(os.path.exists(os.path.join(class_dir, f)) for f in required_files):
                        self.samples.append({
                            'case': case_dir,
                            'class_id': cls_id,
                            'class_dir': class_dir,
                            'case_path': case_path,
                        })
        
        # Train/Val split
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * (1 - val_ratio))
        
        if split == 'train':
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        else:  # val
            self.samples = [self.samples[i] for i in indices[split_idx:]]
        
        print(f"[CardiacPerClass] {split}: {len(self.samples)} samples "
              f"(classes: {self.classes})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        class_dir = sample['class_dir']
        
        try:
            # Load data
            surface_pts = np.load(os.path.join(class_dir, 'surface_pts.npy'))
            near_pts = np.load(os.path.join(class_dir, 'near_pts.npy'))
            near_sdf = np.load(os.path.join(class_dir, 'near_sdf.npy'))
            vol_pts = np.load(os.path.join(class_dir, 'vol_pts.npy'))
            vol_sdf = np.load(os.path.join(class_dir, 'vol_sdf.npy'))
            
            # Flatten SDF if needed (N, 1) -> (N,)
            near_sdf = near_sdf.flatten()
            vol_sdf = vol_sdf.flatten()
            
        except Exception as e:
            print(f"Error loading {class_dir}: {e}")
            # Return a random other sample
            return self.__getitem__(np.random.randint(len(self)))
        
        # Sample surface points
        if surface_pts.shape[0] >= self.surface_size:
            ind = np.random.choice(surface_pts.shape[0], self.surface_size, replace=False)
        else:
            ind = np.random.choice(surface_pts.shape[0], self.surface_size, replace=True)
        surface = surface_pts[ind]
        
        # Sample SDF points (balanced sampling)
        half_size = self.sdf_size // 2
        
        # Volume points: balanced inside/outside
        vol_inside_mask = vol_sdf < 0
        vol_outside_mask = ~vol_inside_mask
        
        vol_inside_idx = np.where(vol_inside_mask)[0]
        vol_outside_idx = np.where(vol_outside_mask)[0]
        
        if len(vol_inside_idx) >= half_size and len(vol_outside_idx) >= half_size:
            inside_choice = np.random.choice(vol_inside_idx, half_size, replace=False)
            outside_choice = np.random.choice(vol_outside_idx, half_size, replace=False)
        else:
            # Fallback: sample with replacement
            inside_choice = np.random.choice(vol_inside_idx, half_size, replace=True) if len(vol_inside_idx) > 0 else np.array([], dtype=int)
            outside_choice = np.random.choice(vol_outside_idx, half_size, replace=True) if len(vol_outside_idx) > 0 else np.array([], dtype=int)
        
        vol_idx = np.concatenate([inside_choice, outside_choice])
        sampled_vol_pts = vol_pts[vol_idx]
        sampled_vol_sdf = vol_sdf[vol_idx]
        
        # Near points: random sample
        near_idx = np.random.choice(near_pts.shape[0], self.sdf_size, replace=False)
        sampled_near_pts = near_pts[near_idx]
        sampled_near_sdf = near_sdf[near_idx]
        
        # Convert to tensors
        surface = torch.from_numpy(surface).float()
        
        if self.split == 'train':
            # Combine vol and near points for training
            points = np.concatenate([sampled_vol_pts, sampled_near_pts], axis=0)
            sdf = np.concatenate([sampled_vol_sdf, sampled_near_sdf], axis=0)
        else:
            # Validation: use only near points (like original VecSetX)
            points = sampled_near_pts
            sdf = sampled_near_sdf
        
        points = torch.from_numpy(points).float()
        sdf = torch.from_numpy(sdf).float()
        
        # Apply augmentation (random rotation)
        if self.augment:
            surface, points = self._random_rotation(surface, points)
        
        # Apply transform if provided
        if self.transform:
            surface, points = self.transform(surface, points)
        
        # Return occupancy instead of SDF if requested
        if not self.return_sdf:
            sdf = (sdf < 0).float()
        
        return points, sdf, surface, sample['case'], sample['class_id']
    
    def _random_rotation(self, surface, points):
        """Apply random rotation augmentation (same as original VecSetX)."""
        # Random axis permutation
        perm = torch.randperm(3)
        points = points[:, perm]
        surface = surface[:, perm]
        
        # Random sign flip
        negative = torch.randint(2, size=(3,)) * 2 - 1
        points = points * negative[None].float()
        surface = surface * negative[None].float()
        
        # Random rotation
        roll = torch.randn(1)
        yaw = torch.randn(1)
        pitch = torch.randn(1)
        
        tensor_0 = torch.zeros(1)
        tensor_1 = torch.ones(1)
        
        RX = torch.stack([
            torch.stack([tensor_1, tensor_0, tensor_0]),
            torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
            torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])
        ]).reshape(3, 3)
        
        RY = torch.stack([
            torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
            torch.stack([tensor_0, tensor_1, tensor_0]),
            torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])
        ]).reshape(3, 3)
        
        RZ = torch.stack([
            torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
            torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
            torch.stack([tensor_0, tensor_0, tensor_1])
        ]).reshape(3, 3)
        
        R = torch.mm(RZ, RY)
        R = torch.mm(R, RX)
        
        points = torch.mm(points, R).detach()
        surface = torch.mm(surface, R).detach()
        
        return surface, points


def create_cardiac_dataloaders(
    dataset_folder,
    classes=None,
    batch_size=16,
    surface_size=8192,
    sdf_size=4096,
    num_workers=4,
    val_ratio=0.1,
    seed=42,
):
    """
    Create train and validation dataloaders for cardiac data.
    
    Args:
        dataset_folder: Path to the dataset directory
        classes: List of class IDs to include (e.g., [1] for Myocardium only)
        batch_size: Batch size
        surface_size: Number of surface points
        sdf_size: Number of SDF query points
        num_workers: Number of data loading workers
        val_ratio: Ratio for validation set
        seed: Random seed
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = CardiacPerClass(
        dataset_folder=dataset_folder,
        split='train',
        classes=classes,
        surface_size=surface_size,
        sdf_size=sdf_size,
        augment=True,
        val_ratio=val_ratio,
        seed=seed,
    )
    
    val_dataset = CardiacPerClass(
        dataset_folder=dataset_folder,
        split='val',
        classes=classes,
        surface_size=surface_size,
        sdf_size=sdf_size,
        augment=False,
        val_ratio=val_ratio,
        seed=seed,
    )
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader
