import os
from pathlib import Path
import shutil
import torch
from torch.utils.data import Dataset
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


class AhmedMLDataset(Dataset):
    """Dataset for AhmedML data from N. Ashton (https://huggingface.co/datasets/neashton/ahmedml)
    
    Args:
        saved_folder (str): Path to the folder where the data is stored.
        if_test (bool): If True, use the test split. Otherwise, use the training split.
        geometry_points (int): Number of geometry points to sample.
        surface_points (int): Number of surface points to sample.
        volume_points (int): Number of volume points to sample.
        copy_to_node (bool): If True, copy the data to the node where the training is running to make loading faster.
                             If you want to use this, make sure that the correct path is set in the copy_data_to_node 
                             function.
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
                 copy_to_node=True,
                 prepare_data=False,
                 fast_approx_sampling=False,
                 scale_positions=False):
        print(f"Using {geometry_points} geometry points, {surface_points} surface points, and {volume_points} volume points.")
        
        if scale_positions:
            self.min_pos = torch.tensor([-4.0, -4.0, -4.0])
            self.max_pos = torch.tensor([6.0, 6.0, 6.0])
        else:
            self.min_pos = torch.tensor([-4.0, -1.0, 0.0])
            self.max_pos = torch.tensor([6.0, 1.0, 1.4])
            
        self.geometry_points = geometry_points
        self.surface_points = surface_points
        self.volume_points = volume_points
        self.fast_approx_sampling = fast_approx_sampling
        self.file_path = os.path.abspath(saved_folder)
        
        # Load all ids from the folders
        ids =  [int(f.name.split('_')[1]) for f in os.scandir(self.file_path) if f.is_dir()]
        ids.sort()
        ids = np.array(ids)
        self.all_ids = ids
        
        # Test and train split with 100 random samples for testing and 400 for training
        self.training_ids = ids = [244, 488, 431, 433, 312,  63, 492, 490, 293, 347, 207, 499,  43,
                    319, 108, 247, 402, 327, 362, 448, 259, 233, 163, 489,  37, 217,
                    111,  23, 369, 422, 156, 457, 152, 401, 416, 423, 381, 361, 317,
                    296, 264, 418, 275, 456,  87, 446, 394, 370, 304, 261, 421, 313,
                    411, 484, 213,  78,  81, 117, 159, 146, 105, 363, 222, 426, 430,
                    497, 481, 424, 180, 204, 101, 151, 354, 126, 276,  18, 188, 331,
                    72, 473, 169,  34, 150, 212, 260,  38, 280, 410, 149,  25, 305,
                    350, 294, 283,  13,  73, 187,  90, 199, 374, 494, 245, 427, 214,
                    500, 367,  30,  57, 353, 356,  27, 205, 392, 257, 495, 491, 297,
                    398,  94, 439,  36,  19, 198, 172, 148,  55,  68,  24, 475, 325,
                    69,  20, 485, 324, 342, 432, 443,  49, 124, 471,  79, 493, 359,
                    123, 120, 173, 292, 440,  77, 447, 144,  12, 143,  46, 348, 220,
                    102, 409, 406, 241,  29, 278, 238,   7, 291, 355,  21, 116,  39,
                    2, 351,  16,  48,  11, 231, 459, 239, 282,  51, 318, 142, 132,
                    279, 174, 290, 139, 479,  71, 309, 141, 311,  76,   5, 480, 208,
                    344, 496, 249,  28, 118, 301, 253, 197, 399, 133, 190, 107,  60,
                    477,  64, 386,   6, 442, 196, 339, 341, 435, 189, 271,  89, 250,
                    255, 221, 300, 389, 272, 400,  96, 428, 186, 441, 288,  41, 333,
                    114, 474, 202, 127, 270, 455, 498, 242, 230, 472, 360, 415,  84,
                    375, 366, 286, 378, 113, 391, 170, 267,  45, 228, 444,  42, 263,
                    437,  44, 384, 109,   9, 100, 167, 408, 352, 211,  95, 243, 329,
                    164, 287,  26, 458, 171, 453,  32,  17,  59, 383, 158,  54,  14,
                    425, 157, 110, 147,  91, 119, 262,  70, 340, 252, 467, 320, 256,
                    326, 137, 349, 226, 454, 469, 210, 224, 206, 162,  88,  65, 112,
                    373, 377, 251, 299, 396, 165,  33, 281, 155, 468,  75, 181, 122,
                    225, 338, 121, 376, 450, 420, 405, 368, 178, 478, 168, 135,  61,
                    306,  56, 323,  80, 154, 388,   4, 200, 273, 240, 461, 413, 379,
                    176, 404, 254, 463,  93,  52, 307, 192, 232, 419, 161, 372, 179,
                    195, 140, 436, 115, 358, 246,  22, 235, 285, 308, 185, 131, 466,
                    62, 407, 482, 215,  47, 166, 337, 104, 277, 343, 483, 470, 487,
                    183, 223, 201, 138, 322, 289, 103, 465, 438, 216]
        self.test_ids = [175, 476, 414, 258, 434,  58,  85, 385, 390, 303,  66,  35, 403,
                    97,  92, 295, 219, 274, 486,  86, 335, 332, 134, 328, 321, 236,
                    310,  67, 136,  99, 445, 395, 130, 106,  82, 464, 357, 364, 184,
                    460, 218, 397, 177, 336,   1, 269, 125,  15, 449, 365, 314,  83,
                    193, 194, 237,  40, 191, 266,  53, 129, 462, 227, 203,   3, 330,
                    334, 160, 380, 268, 382, 412,  74, 429, 234, 128, 417, 371, 209,
                    302, 265,  98, 345, 387, 145, 182, 452, 346, 284, 248, 316, 393,
                    153, 298, 315, 229,   8,  50,  31, 451,  10]
        
        if if_test:
            self.data = self.test_ids
        else:
            self.data = self.training_ids
        
        if prepare_data:
            print("Precompute numpy arrays...")
            self.precompute_numpy_arrays()
            print("Computing stats...")
            self.compute_stats()
            
        if copy_to_node:
            user = os.getenv("USER")
            self.copy_data_to_node(f"/data/scratch/{user}/data/ahmedml/")
            
        # Load the statistics for normalization
        self.load_stats()

    def copy_data_to_node(self, path, force_copy=False):
        """Copy the data to the node where the training is running to make loading faster."""
        
        if not os.path.exists(path) or force_copy:
            print(f"Creating directory {path}")
            os.makedirs(path, exist_ok=True)
            for id in self.all_ids:
                src_folder = os.path.join(self.file_path, f"run_{id}")
                dst_folder = os.path.join(path, f"run_{id}")
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
                for file in os.listdir(src_folder):
                    if file.endswith(".npy"):
                        src_file = os.path.join(src_folder, file)
                        dst_file = os.path.join(dst_folder, file)
                        if not os.path.exists(dst_file):
                            shutil.copy(src_file, dst_file)
            
            # Copy stats files if they exist
            stats_files = ["volume_stats.npy", "surface_stats.npy", "position_stats.npy"]
            for stats_file in stats_files:
                src_file = os.path.join(self.file_path, stats_file)
                dst_file = os.path.join(path, stats_file)
                if os.path.exists(src_file) and not os.path.exists(dst_file):
                    shutil.copy(src_file, dst_file)
        else:
            print(f"Data already copied to {path}, skipping copy step.")
        
        # Update the file path to the copied data
        self.file_path = path
    
    def precompute_numpy_arrays(self):
        """Load the data to precompute the numpy arrays for faster loading later."""
        
        for id in self.all_ids:
            print(f"Precompute numpy array for sample {id}")
            folder = os.path.join(self.file_path, f"run_{id}")
            _ = self.get_surface_mesh(folder, id)
            _ = self.get_surface_data(folder, id)
            _ = self.get_volume_data(folder, id)
    
    def load_stats(self):
        """Load the precomputed mean and std of the dataset for normalization."""
        vol_stats_file = Path(os.path.join(self.file_path, "volume_stats.npy"))
        surf_stats_file = Path(os.path.join(self.file_path, "surface_stats.npy"))
        pos_stats_file = Path(os.path.join(self.file_path, "position_stats.npy"))

        if vol_stats_file.is_file() and surf_stats_file.is_file() and pos_stats_file.is_file():
            print("Loading stats")
            # Volume data
            data = np.load(vol_stats_file)
            self.mean_vol_data = torch.tensor(data[0])[:3]  # only velocity mean
            self.std_vol_data = torch.tensor(data[1])[:3]    # only velocity std
            
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

        vol_data_sum = torch.zeros((6,), dtype=torch.float32)
        vol_data_squared_sum = torch.zeros((6,), dtype=torch.float32)
        vol_data_count = 0
        
        # Iterate over training samples
        for id in self.training_ids:
            print(f"Computing stats for sample {id}")
            folder = os.path.join(self.file_path, f"run_{id}")
            geo_mesh = self.get_surface_mesh(folder, id)
            surf_mesh, surf_data = self.get_surface_data(folder, id)
            vol_mesh, vol_data = self.get_volume_data(folder, id)
            
            for d in range(3):
                max_pos[d] = max(max_pos[d], geo_mesh[:, d].max().item(), surf_mesh[:, d].max().item(), vol_mesh[:, d].max().item())
                min_pos[d] = min(min_pos[d], geo_mesh[:, d].min().item(), surf_mesh[:, d].min().item(), vol_mesh[:, d].min().item())

            surf_data_sum += surf_data.sum(dim=0)
            vol_data_sum += vol_data.sum(dim=0)

            surf_data_squared_sum += (surf_data ** 2).sum(dim=0)
            vol_data_squared_sum += (vol_data ** 2).sum(dim=0)

            surf_data_count += surf_data.shape[0]
            vol_data_count += vol_data.shape[0]
        
        std = lambda sum, squared_sum, count: torch.sqrt((squared_sum - ((sum ** 2) / count)) / (count-1))
        
        self.mean_surf_data = surf_data_sum / surf_data_count
        self.std_surf_data = std(surf_data_sum, surf_data_squared_sum, surf_data_count)
        
        self.mean_vol_data = vol_data_sum / vol_data_count
        self.std_vol_data = std(vol_data_sum, vol_data_squared_sum, vol_data_count)
        
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

    def get_surface_mesh(self, folder, idx):
        if not os.path.isfile(os.path.join(folder, "body.npy")):
            reader = vtk.vtkSTLReader()
            reader.SetFileName(os.path.join(folder, f"ahmed_{self.data[idx]}.stl"))
            reader.Update()

            # Get the geometry as vtkPolyData
            polydata = reader.GetOutput()

            # Extract point positions
            points = polydata.GetPoints()
            positions = torch.tensor(vtk_to_numpy(points.GetData()), dtype=torch.float32)
            
            # Save the positions to a numpy file for future use
            np.save(os.path.join(folder, "body.npy"), positions.numpy())
        else:
            # Load the positions from the saved numpy file
            positions = torch.tensor(np.load(os.path.join(folder, "body.npy")), dtype=torch.float32)
        
        return positions

    def get_surface_data(self, folder, idx):
        if not os.path.isfile(os.path.join(folder, "surface.npy")):
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(os.path.join(folder, f"boundary_{self.data[idx]}.vtp"))
            reader.Update()

            polydata = reader.GetOutput()
            cell_centers_filter = vtk.vtkCellCenters()
            cell_centers_filter.SetInputData(polydata)
            cell_centers_filter.Update()
            cell_centers = cell_centers_filter.GetOutput()
            points_centers = cell_centers.GetPoints()
            mesh = torch.tensor(vtk_to_numpy(points_centers.GetData()), dtype=torch.float32)
            
            cell_data = polydata.GetCellData()
            data = torch.tensor(vtk_to_numpy(cell_data.GetArray("pMean")), dtype=torch.float32)
            
            # Save the mesh and pressure data to a numpy file for future use
            np.save(os.path.join(folder, "surface.npy"), mesh.numpy())
            np.save(os.path.join(folder, "pressure.npy"), data.numpy())
        else:
            # Load the mesh and pressure data from the saved numpy files
            mesh = torch.tensor(np.load(os.path.join(folder, "surface.npy")), dtype=torch.float32)
            data = torch.tensor(np.load(os.path.join(folder, "pressure.npy")), dtype=torch.float32)
        
        return mesh, data[..., None]
    
    def get_volume_data(self, folder, idx):
        if not os.path.isfile(os.path.join(folder, "volume_data.npy")):
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(os.path.join(folder, f"volume_{self.data[idx]}.vtu"))
            reader.Update()

            polydata = reader.GetOutput()
            cell_centers_filter = vtk.vtkCellCenters()
            cell_centers_filter.SetInputData(polydata)
            cell_centers_filter.Update()
            cell_centers = cell_centers_filter.GetOutput()
            points_centers = cell_centers.GetPoints()
            mesh = torch.tensor(vtk_to_numpy(points_centers.GetData()), dtype=torch.float32)
            
            cell_data = polydata.GetCellData()
            velocity = torch.tensor(vtk_to_numpy(cell_data.GetArray("UMean")), dtype=torch.float32)
            vorticity = torch.tensor(vtk_to_numpy(cell_data.GetArray("vorticityMean")), dtype=torch.float32)
            data = torch.tensor(np.concat((velocity.numpy(), vorticity.numpy()), axis=1))

            # Save the mesh and velocity data to a numpy file for future use
            np.save(os.path.join(folder, "volume.npy"), mesh.numpy())
            np.save(os.path.join(folder, "volume_data.npy"), data)
        else:
            # Load the mesh and velocity data from the saved numpy files
            mesh = torch.tensor(np.load(os.path.join(folder, "volume.npy")), dtype=torch.float32)
            data = torch.tensor(np.load(os.path.join(folder, "volume_data.npy")), dtype=torch.float32)
        
        return mesh, data
    
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
        folder = os.path.join(self.file_path, f"run_{self.data[idx]}")
        geo_mesh = self.get_surface_mesh(folder, idx)
        surf_mesh, surf_data = self.get_surface_data(folder, idx)
        vol_mesh, vol_data = self.get_volume_data(folder, idx)
        
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
        
        # Consider only velocity in the volume for now
        vol_data = vol_data[:, :3]
        
        vol_mesh = (vol_mesh[vol_points, :] - self.min_pos) / (self.max_pos - self.min_pos)
        vol_data = (vol_data[vol_points, :] - self.mean_vol_data) / self.std_vol_data
        
        return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data
