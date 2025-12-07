
import os

import torch
from torch.utils import data

import numpy as np

import csv

import trimesh

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
        ):
        
        self.surface_size = surface_size

        self.transform = transform
        self.sdf_sampling = sdf_sampling
        self.sdf_size = sdf_size
        self.split = split

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
                surface = data['surface_points']
                if 'surface_labels' in data:
                    surface_labels = data['surface_labels']
                else:
                    surface_labels = None
                
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

        if self.surface_sampling:
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

        # surface_normals = torch.from_numpy(surface_normals).float()

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

        return points, sdf, surface, self.models[idx][1], npz_path#, surface_normals

    def __len__(self):
        return len(self.models)
