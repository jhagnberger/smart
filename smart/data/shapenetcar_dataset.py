import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


class ShapeNetCarDataset(Dataset):
    """Dataset for ShapeNetCar data from N. Umetani (http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip)
    
    Args:
        saved_folder (str): Path to the folder where the data is stored.
        if_test (bool): If True, use the test split. Otherwise, use the training split.
        geometry_points (int): Number of geometry points to sample.
        surface_points (int): Number of surface points to sample.
        volume_points (int): Number of volume points to sample.
        prepare_data (bool): If True, precompute numpy arrays and compute statistics for normalization.
        fast_approx_sampling (bool): If True, use fast but approximate sampling of points (may include duplicates, but 
                                     that is unlikely if dataset is large enough).
                                     If False, use slower but exact sampling without duplicates.
        scale_positions (bool): If True, scale the positions uniformly in all directions to avoid distortions.
    """
    
    def __init__(self,
                 saved_folder='../data/',
                 if_test=False,
                 geometry_points=65536,
                 surface_points=65536,
                 volume_points=65536,
                 prepare_data=False,
                 fast_approx_sampling=False,
                 scale_positions=False):
        print(f"Using {geometry_points} geometry points, {surface_points} surface points, and {volume_points} volume points.")
        
        if scale_positions:
            self.min_pos = torch.tensor([-4.5, -4.5, -4.5])
            self.max_pos = torch.tensor([6.0, 6.0, 6.0])
        else:
            self.min_pos = torch.tensor([-2.0, -1.0, -4.5])
            self.max_pos = torch.tensor([2.0, 4.5, 6.0])
        
        self.geometry_points = geometry_points
        self.surface_points = surface_points
        self.volume_points = volume_points
        self.fast_approx_sampling = fast_approx_sampling
        self.file_path = os.path.abspath(saved_folder)
        
        # Load all samples from the folders
        self.all_ids = self.get_samples()
        
        # We use the first 100 samples (aka param0) for testing
        self.training_ids = self.all_ids[100:]
        self.test_ids = self.all_ids[:100]
        
        if if_test:
            self.data = self.test_ids
        else:
            self.data = self.training_ids
        
        if prepare_data:
            print("Precompute numpy arrays...")
            self.precompute_numpy_arrays()
            print("Computing stats...")
            self.compute_stats()
            
        # Load the data into RAM
        self.surface_meshes, self.surf_data, self.vol_meshes, self.vol_data = self.load_data()
            
        # Load the statistics for normalization
        self.load_stats()

    def get_samples(self):
        """Get all sample IDs from the dataset folder."""
        samples = []
        for dir in [f"param{i}" for i in range(9)]:
            files = os.listdir(os.path.join(self.file_path, dir))
            for file in files:
                path = os.path.join(self.file_path, os.path.join(dir, file))
                if os.path.isdir(path) and os.path.isfile(os.path.join(path, "quadpress_smpl.vtk")) and os.path.isfile(os.path.join(path, "hexvelo_smpl.vtk")):
                    samples.append(os.path.join(dir, file))
        return samples

    def copy_data_to_node(self, path):
        """Copy the data to the node where the training is running to make loading faster."""
        
        raise NotImplementedError("Data copying to node is not used since the data is loaded directly into RAM.")
    
    def precompute_numpy_arrays(self):
        """Load the data to precompute the numpy arrays for faster loading later."""
        
        for id in self.all_ids:
            print(f"Precompute numpy array for sample {id}")
            _ = self.get_surface_data(id)
            _ = self.get_volume_data(id)
    
    def load_stats(self):
        """Load the precomputed mean and std of the dataset for normalization."""
        vol_stats_file = Path(os.path.join(self.file_path, "volume_stats.npy"))
        surf_stats_file = Path(os.path.join(self.file_path, "surface_stats.npy"))
        pos_stats_file = Path(os.path.join(self.file_path, "position_stats.npy"))

        if vol_stats_file.is_file() and surf_stats_file.is_file() and pos_stats_file.is_file():
            print("Loading stats")
            # Volume data
            data = np.load(vol_stats_file)
            self.mean_vol_data = torch.tensor(data[0])
            self.std_vol_data = torch.tensor(data[1])
            
            # Surface data
            data = np.load(surf_stats_file)
            self.mean_surf_data = torch.tensor(data[0])
            self.std_surf_data = torch.tensor(data[1])
            
            # Coordinates
            # data = np.load(pos_stats_file)
            # self.min_pos = torch.tensor(data[0])
            # self.max_pos = torch.tensor(data[1])
            
            print(f"Average surface: {self.mean_surf_data}")
            print(f"Average volume: {self.mean_vol_data}")
            print(f"Std surface: {self.std_surf_data}")
            print(f"Std volume: {self.std_vol_data}")
            print(f"Min position: {self.min_pos}")
            print(f"Max position: {self.max_pos}")
        else:
            raise FileNotFoundError("Stats files not found, please compute them first by setting prepare_data=True.")
        
    def compute_stats(self):
        """Iteratively compute the mean and std of the dataset for normalization."""

        min_pos = torch.full((3,), np.inf, dtype=torch.float32)
        max_pos = torch.full((3,), -np.inf, dtype=torch.float32)

        surf_data_sum = torch.zeros((1,), dtype=torch.float32)
        surf_data_squared_sum = torch.zeros((1,), dtype=torch.float32)
        surf_data_count = 0

        velo_sum = torch.zeros((3,), dtype=torch.float32)
        velo_squared_sum = torch.zeros((3,), dtype=torch.float32)
        velo_count = 0
        
        # Iterate over training samples
        for id in self.training_ids:
            surf_mesh, pressure = self.get_surface_data(id)
            vol_mesh, velo = self.get_volume_data(id)
            
            for d in range(3):
                max_pos[d] = max(max_pos[d], surf_mesh[:, d].max().item(), vol_mesh[:, d].max().item())
                min_pos[d] = min(min_pos[d], surf_mesh[:, d].min().item(), vol_mesh[:, d].min().item())

            surf_data_sum += pressure.sum(dim=0)
            velo_sum += velo.sum(dim=0)

            surf_data_squared_sum += (pressure ** 2).sum(dim=0)
            velo_squared_sum += (velo ** 2).sum(dim=0)

            surf_data_count += pressure.shape[0]
            velo_count += velo.shape[0]
        
        std = lambda sum, squared_sum, count: torch.sqrt((squared_sum - ((sum ** 2) / count)) / (count-1))
        
        self.mean_surf_data = surf_data_sum / surf_data_count
        self.std_surf_data = std(surf_data_sum, surf_data_squared_sum, surf_data_count)
        
        self.mean_vol_data = velo_sum / velo_count
        self.std_vol_data = std(velo_sum, velo_squared_sum, velo_count)
        
        self.min_pos = min_pos
        self.max_pos = max_pos

        # Save the stats to a file for future use
        vol_stats_file = Path(os.path.join(self.file_path, "volume_stats.npy"))
        surf_stats_file = Path(os.path.join(self.file_path, "surface_stats.npy"))
        pos_stats_file = Path(os.path.join(self.file_path, "position_stats.npy"))
        np.save(surf_stats_file, np.array([self.mean_surf_data, self.std_surf_data]))
        np.save(vol_stats_file, np.array([self.mean_vol_data, self.std_vol_data]))
        np.save(pos_stats_file, np.array([self.min_pos, self.max_pos]))

        print(f"Average surface: {self.mean_surf_data}")
        print(f"Average volume: {self.mean_vol_data}")
        print(f"Std surface: {self.std_surf_data}")
        print(f"Std volume: {self.std_vol_data}")
        print(f"Min position: {self.min_pos}")
        print(f"Max position: {self.max_pos}")

    def get_surface_data(self, sample):
        folder = os.path.join(self.file_path, sample)
        if not os.path.isfile(os.path.join(folder, "surface.npy")):
            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(os.path.join(folder, "quadpress_smpl.vtk"))
            reader.Update()

            polydata = reader.GetOutput()
            mesh = torch.tensor(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=torch.float32)
            data = torch.tensor(vtk_to_numpy(polydata.GetPointData().GetScalars()), dtype=torch.float32)
            
            # Save the mesh and pressure data to a numpy file for future use
            np.save(os.path.join(folder, "surface.npy"), mesh.numpy())
            np.save(os.path.join(folder, "pressure.npy"), data.numpy())
        else:
            # Load the mesh and pressure data from the saved numpy files
            mesh = torch.tensor(np.load(os.path.join(folder, "surface.npy")), dtype=torch.float32)
            data = torch.tensor(np.load(os.path.join(folder, "pressure.npy")), dtype=torch.float32)
         
        return mesh, data[..., None]
    
    def get_volume_data(self, sample):
        folder = os.path.join(self.file_path, sample)
        if not os.path.isfile(os.path.join(folder, "volume_data.npy")):
            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(os.path.join(folder, "hexvelo_smpl.vtk"))
            reader.Update()

            polydata = reader.GetOutput()
            mesh = torch.tensor(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=torch.float32)
            data = torch.tensor(vtk_to_numpy(polydata.GetPointData().GetVectors()), dtype=torch.float32)
            
            # Save the mesh and velocity data to a numpy file for future use
            np.save(os.path.join(folder, "volume.npy"), mesh.numpy())
            np.save(os.path.join(folder, "volume_data.npy"), data)
        else:
            # Load the mesh and velocity data from the saved numpy files
            mesh = torch.tensor(np.load(os.path.join(folder, "volume.npy")), dtype=torch.float32)
            data = torch.tensor(np.load(os.path.join(folder, "volume_data.npy")), dtype=torch.float32)
        
        return mesh, data
    
    def load_data(self):
        """Load all data into RAM for faster access during training."""
        
        surf_meshes = []
        vol_meshes = []
        surf_data = []
        vol_data = []
        for id in self.data:
            surf_mesh, surf_data_ = self.get_surface_data(id)
            vol_mesh, vol_data_ = self.get_volume_data(id)

            surf_meshes.append(surf_mesh)
            surf_data.append(surf_data_)
            vol_meshes.append(vol_mesh)
            vol_data.append(vol_data_)

        surf_meshes = torch.stack(surf_meshes)
        surf_data = torch.stack(surf_data)
        vol_meshes = torch.stack(vol_meshes)
        vol_data = torch.stack(vol_data)
        
        return surf_meshes, surf_data, vol_meshes, vol_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retrieves a sample for a given index with the geometry, surface mesh and data, and volume mesh and data.
        
        Args:
            idx (int): Index of the data sample to retrieve.
            
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The geometry mesh of the data sample as tensor with shape (number geo points, 3).
                - torch.Tensor: The surface mesh of the data sample as tensor with shape (number surf points, 3).
                - torch.Tensor: The surface data (pressure) of the data sample as tensor with shape (number surf points, 1).
                - torch.Tensor: The volume mesh of the data sample as tensor with shape (number vol points, 3).
                - torch.Tensor: The volume data of the data sample as tensor with shape (number vol points, 3).
        """
        # Load the data for the given index
        geo_mesh = self.surface_meshes[idx, ...]
        surf_mesh, surf_data = self.surface_meshes[idx, ...], self.surf_data[idx, ...]
        vol_mesh, vol_data = self.vol_meshes[idx, ...], self.vol_data[idx, ...]

        # We can subsample the data to reduce the resolution
        if self.geometry_points > 0:
            if not self.fast_approx_sampling:
                # This is slow but gives unique points
                geo_points = torch.randperm(geo_mesh.shape[0])[:self.geometry_points]
            else:
                # This is fast but may give duplicate points
                geo_points = torch.randint(0, geo_mesh.shape[0], (self.geometry_points,))
        else:
            geo_points = torch.arange(geo_mesh.shape[0])
        geo_mesh = (geo_mesh[geo_points, :] - self.min_pos) / (self.max_pos - self.min_pos)

        if self.surface_points > 0:
            if not self.fast_approx_sampling:
                surface_points = torch.randperm(surf_mesh.shape[0])[:self.surface_points]
            else:
                surface_points = torch.randint(0, surf_mesh.shape[0], (self.surface_points,))
        else:
            surface_points = torch.arange(surf_mesh.shape[0])
        surf_mesh = (surf_mesh[surface_points, :] - self.min_pos) / (self.max_pos - self.min_pos)
        surf_data = (surf_data[surface_points, :] - self.mean_surf_data) / self.std_surf_data

        if self.volume_points > 0:
            if not self.fast_approx_sampling:
                vol_points = torch.randperm(vol_mesh.shape[0])[:self.volume_points]
            else:
                vol_points = torch.randint(0, vol_mesh.shape[0], (self.volume_points,))
        else:
            vol_points = vol_points = torch.arange(vol_mesh.shape[0])
        vol_mesh = (vol_mesh[vol_points, :] - self.min_pos) / (self.max_pos - self.min_pos)
        vol_data = (vol_data[vol_points, :] - self.mean_vol_data) / self.std_vol_data
        
        # Consider only velocity in the volume for now
        velocity = vol_data[:, :3]
        
        return geo_mesh, surf_mesh, surf_data, vol_mesh, velocity
