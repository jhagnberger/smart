import os
from pathlib import Path
import shutil
import torch
from torch.utils.data import Dataset
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


class ShiftSUVDataset(Dataset):
    """Dataset for SHIFT-SUV data from Luminary Cloud (https://huggingface.co/datasets/luminary-shift/SUV)
    
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
                 fast_approx_sampling=True,
                 scale_positions=False):
        print(f"Using {geometry_points} geometry points, {surface_points} surface points, and {volume_points} volume points.")
        
        if scale_positions:
            self.min_pos = torch.tensor([-40.0, -40.0, -40.0])
            self.max_pos = torch.tensor([80.0, 80.0, 80.0])
        else:
            self.min_pos = torch.tensor([-40.0, -25.0, -5.0])
            self.max_pos = torch.tensor([80.0, 25.0, 25.0])
        
        self.geometry_points = geometry_points
        self.surface_points = surface_points
        self.volume_points = volume_points
        self.fast_approx_sampling = fast_approx_sampling
        self.file_path = os.path.abspath(saved_folder)
        
        # load all ids from the folders
        ids =  [int(f.name.split('_')[1]) for f in os.scandir(self.file_path) if f.is_dir() and any(os.scandir(f.path))]
        ids.sort()
        ids = np.array(ids)
        self.all_ids = ids
        
        # test and train split
        # we had to remove file 845 because the files were corrupted
        self.training_ids = [  38,   94,    2,  310,  964,   10,  278,  467,  901,   58,  910,
                        91,  364,  979,  378,  741,  275,  755,  639,  611,  119,
                        673,   14,   74,  753,  563,  446,  585,  981,  617,  416,  958,
                        824,  804,  514,  780,  423,  844,   76,  904,  102,  955,  793,
                        193,  916,  859,  198,   96,  601,  339,  492,  365,  874,  123,
                        473,   30,  870,  731,  873,  117,   35,  382,  302,  878,  360,
                        274,  654,  430,  627,  776,  509,  184,  546,  631,  661,
                        146,  653,  159,  720,  963,  628,   87,  691,  880,  529,  884,
                        115,  386,  614,  396,  244,  799,   44,  700,  838,  761,  201,
                        81,  410,  414,  291,  506,  949,  659,  329,  604,  586,  641,
                        950,  619,  172,  153,  620,  144,   95,  164,   69,  926,  822,
                        452,  277,  717,  260,   36,  539,  648,  533,  729,  238,   12,
                        93,  875,  823,  312,    8,  128,  232,  637,  311,  178,  133,
                        869,  130,  406,  889,  531,  319,  341,  814,   21,  985,  970,
                        724,  989,  674,   33,  412,  734,  843,  819,  258,  624,  779,
                        942,  106,  583,  293,  202,  792,  552,  695,  994,  186,  588,
                        262,  784,  758,  809,  321,   80,  472,  292,  710,  806,  114,
                        449,  577,  760,  485,  306,   48,  223,  733,  746,  770,  933,
                        124,  466,  402,  862,  390,  775,  664,  344,  129,  345,  404,
                        263,  993,  833,  666,  890,   89,  395,  135,  676,  217,  682,
                        853,  962,  920,  848,  919,  187,  107,  820,  735,  459,  497,
                        104,  821,  255,  505,  195,  185,  669,  408,  670,  941,  751,
                        221,  764,  574,  798,  181,  562,   71,  211,  110,  163,  646,
                        177,  487,  456,  632,   47,  865,  180,  645,  122,  542,  376,
                        366,  857,  783,  983,   82,  737,  988,  782,  852,  747,  663,
                        301,   41,  817,  934,  284,  336,   72,  296,  353,   40,  837,
                        668,  399,  513,  105,  998,  165,  281,  788,  743,  547,   55,
                        787,  450,  326,  883,  557,  507,  699,  609,  704,  503,  417,
                        272,  273,  160,  161,  599,  945,  561,  560,  527,  537,  728,
                        544,  299,  173,  921,  842,  943,  462,  892,  401,  987,  254,
                        584,  931,  765,  858,   45,  313,  966,  832,   23,  683,  675,
                        567,  555,   61,   90,  980,  914,  351,  481,  810,  204,  991,
                        464,   53,  536,  196,  638,  621,  109,  443,  722,  520,  727,
                        309,  222,  623,  362,  521,  667,  657,  595,  243,  879,  355,
                        368,  363,  433,  282,  266,  526,  871,  598,  495,  471,  736,
                        974,   13,  739,  276,  419,  545,  400,  944,  361,  192,  636,
                        388,  212,  228,  665,  236,  436,  694,  522,  893,  831,  126,
                        139,  887,  407,   31,  772,  101,  701,  992,  952,  140,  756,
                        839,    7,  479,  250,  600,  911,  225,  357,  554,  696,  168,
                        707,  518,   59,  708,  689,  209,  233,  316,  460,   83,  428,
                        338,  500,   28,  549,  877,  132,    4,  876,  197,  961,  558,
                        863,  375,   88,  389,  656,  461,  179,  968,  285,  924,  121,
                        322,  502,  152,  812,  688,   34,  252,  227,  429,  917,  556,
                        5,  805,  946,  815,  304,  550,  605,  268,  938,  835,  305,
                        575,  155,  830,  441,   22,  794,  630,  538,  447,  712,  684,
                        613,   97,  451,  269,  287,  930,  431,  828,  971,  335,  256,
                        660,  997,  246,  811,  470,  785,  888,  226,    6,  692,  415,
                        610,  490,  615,  210,  686,  327,  136,  324,  996,  169,  486,
                        957,  457,  156,  403,  984,  103,   85,  774,  640,  493,  524,
                        649,  488,  856,  385,  508,  142,  240,  714,  116,   43,  224,
                        358,  587,  738,   25,  426,  191,  754,  534,  331,  872,  454,
                        413,  242,  347,  719,  690,  954,  207,  391,   27,  960,  748,
                        289,  270,  219,  818,  251,   16,  478,  328,  718,  827,  559,
                        593,  516,  190,  702,  834,   11,   86,  978,   49,  411,  148,
                        188,  716,  730,  235,  332,  607,  572,  422,  145,  342,  629,
                        56,  170,  680,  475,  151,  913,   65,  767,  528,  936,  902,
                        1,  677,  855,  501,  553,  532,   73,  288,  543,  343,  137,
                        350,   24,  569,  986,  237,  425,  742,  762,  597,  264,  977,
                        194,  530,  906,  453,  633,  745,  766,  652,  868,  167,  118,
                        463,    9,   26,  409,  651,  925,  885,  726,  439,  564,  679,
                        867,  602,  972,  947,  307,  940,  618,  565,  113,  218,   32,
                        515,  174,  825,  437,  790,  778,  477,  706,  725,  484,  510,
                        899,  951,  777,  359,  448,   42,  469,  771,  171,   29,  851,
                        732,  769,   79,  149,  157,  230,  582,  662,   98,  956,  935,
                        570,  220,  214,  525,  283,  315,  213,   60,  442,  568,  138,
                        141,  622,  829,  932,  606,  271,  687,  440,  948,   92,  489,
                        229,  206,   75,  797,  371,  866,  199,  374,  922,  864,  634,
                        203,  397,  379,  432,  367,  908,  671,   67,  182,  625,  482,
                        740,  795,  903,  320,  581,  705,  571,   57,  635,  496,  591,
                        143,  370,  594,  369,   70,  713,  721,  846,  711,  318,  162,
                        499,  826,  455,  434,  297,  239,  458,  573,  929,  923,  612,
                        540,  205,  840,  836,  418,  383,  861,  380,  759,   77,  608,
                        517,  807,   52,  494,  773,  286,  757,  261]
        self.test_ids = [937, 377, 348, 973, 918, 644, 802, 215, 424, 398,  39, 498, 596,
                        111, 216, 967, 249, 523, 589, 125, 616, 248,  20, 100, 166, 685,
                        78, 789, 280, 749, 723, 813, 468, 849, 850, 642, 373, 476, 183,
                        580, 483, 882, 703, 480, 131, 257, 999, 154, 982, 898, 939, 381,
                        127, 445, 905, 928, 678, 323,  15, 650,  51, 995, 886, 504, 158,
                        337, 896, 715, 592, 881, 150, 294, 176, 435, 750, 709, 134,  18,
                        265, 786, 245, 854, 579, 247, 808, 112, 566, 796, 267, 953, 392,
                        847, 535, 969, 990, 576, 512,  64, 965, 405, 474, 752, 915, 120,
                        907, 340, 511, 697, 393,  54, 384, 354,  63, 551, 465, 234, 175,
                        768, 578, 108, 860, 897, 590, 912, 816, 200,  50, 801, 349, 519,
                        655, 317, 259,  84, 346, 420, 626, 900,  68,  37, 744, 959, 672,
                        891,  17, 300, 394, 421, 658, 603, 548, 334, 241, 444, 189, 438,
                        99, 975, 803, 841, 352, 314, 303, 781, 800, 147, 253, 290,  46,
                        3, 909, 895, 356, 330, 208, 681, 763, 279, 894, 927, 491,  66,
                        19, 387, 325, 976, 698, 643, 693, 231, 427,  62, 541, 372, 298,
                        647, 308, 295, 791, 333]
        
        if if_test:
            # random 200 samples for testing
            self.data = self.test_ids
        else:
            # random 800 samples for training
            self.data = self.training_ids
        
        if prepare_data:
            print("Precompute numpy arrays...")
            self.precompute_numpy_arrays()
            print("Computing stats...")
            self.compute_stats()
            
        # copy data to the node where the training is running to make loading faster
        if copy_to_node:
            user = os.getenv("USER")
            self.copy_data_to_node(f"/data/scratch/{user}/data/shift-suv/")
        
        # Load or compute the statistics for normalization
        self.load_stats()
    
    def copy_data_to_node(self, path, force_copy=False):
        """Copy the data to the node where the training is running to make loading faster."""
        
        if not os.path.exists(path) or force_copy:
            try:
                print(f"Creating directory {path}")
                os.makedirs(path, exist_ok=True)
            except Exception as exc:
                print(f"Error creating directory {path}: {exc}")
                return

            for id in self.all_ids:
                src_folder = os.path.join(self.file_path, f"run_{id:05d}")
                dst_folder = os.path.join(path, f"run_{id:05d}")
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder, exist_ok=True)
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
            folder = os.path.join(self.file_path, f"run_{id:05d}")
            _ = self.get_surface_mesh(folder, id)
            _ = self.get_surface_data(folder, id)
            _ = self.get_volume_data(folder, id)
            
            # remove large files to save space
            vol_file = os.path.join(folder, f"merged_volumes.vtu")
            if os.path.isfile(vol_file):
                os.remove(vol_file)
                print(f"Removed {vol_file}")
    
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
        
        vol_stats_file = Path(os.path.join(self.file_path, "volume_stats.npy"))
        surf_stats_file = Path(os.path.join(self.file_path, "surface_stats.npy"))
        pos_stats_file = Path(os.path.join(self.file_path, "position_stats.npy"))

        min_pos = torch.full((3,), np.inf, dtype=torch.float32)
        max_pos = torch.full((3,), -np.inf, dtype=torch.float32)

        surf_data_sum = torch.zeros((1,), dtype=torch.float32)
        surf_data_squared_sum = torch.zeros((1,), dtype=torch.float32)
        surf_data_count = 0

        vol_data_sum = torch.zeros((3,), dtype=torch.float32)
        vol_data_squared_sum = torch.zeros((3,), dtype=torch.float32)
        vol_data_count = 0
        
        # Iterate over training samples
        for id in self.training_ids:
            print(f"Computing stats for sample {id}")
            folder = os.path.join(self.file_path, f"run_{id:05d}")
            mesh = self.get_surface_mesh(folder, id)
            surf_mesh, surf_data = self.get_surface_data(folder, id)
            vol_mesh, vol_data = self.get_volume_data(folder, id)
            
            for d in range(3):
                max_pos[d] = max(max_pos[d], mesh[:, d].max().item(), surf_mesh[:, d].max().item(), vol_mesh[:, d].max().item())
                min_pos[d] = min(min_pos[d], mesh[:, d].min().item(), surf_mesh[:, d].min().item(), vol_mesh[:, d].min().item())
            
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
            reader.SetFileName(os.path.join(folder, f"merged_surfaces.stl"))
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
            reader.SetFileName(os.path.join(folder, f"merged_surfaces.vtp"))
            reader.Update()

            polydata = reader.GetOutput()
            cell_centers_filter = vtk.vtkCellCenters()
            cell_centers_filter.SetInputData(polydata)
            cell_centers_filter.Update()
            cell_centers = cell_centers_filter.GetOutput()
            points_centers = cell_centers.GetPoints()
            mesh = torch.tensor(vtk_to_numpy(points_centers.GetData()), dtype=torch.float32)
            
            cell_data = polydata.GetCellData()
            data = torch.tensor(vtk_to_numpy(cell_data.GetArray("pressure_average")), dtype=torch.float32)
            #wall_shear_stress = torch.tensor(vtk_to_numpy(cell_data.GetArray("wall_shear_stress_average")), dtype=torch.float32)
            
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
            reader.SetFileName(os.path.join(folder, f"merged_volumes.vtu"))
            reader.Update()

            polydata = reader.GetOutput()
            points = polydata.GetPoints()
            mesh = torch.tensor(vtk_to_numpy(points.GetData()), dtype=torch.float32)
            
            point_data = polydata.GetPointData()
            velocity = torch.tensor(vtk_to_numpy(point_data.GetArray("velocity_average")), dtype=torch.float32)
            #pressure = torch.tensor(vtk_to_numpy(point_data.GetArray("vorticityMean")), dtype=torch.float32)
            data = torch.tensor(velocity.numpy())

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
        folder = os.path.join(self.file_path, f"run_{self.data[idx]:05d}")
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
        vol_mesh = (vol_mesh[vol_points, :] - self.min_pos) / (self.max_pos - self.min_pos)
        vol_data = (vol_data[vol_points, :] - self.mean_vol_data) / self.std_vol_data
        
        # Consider only velocity in the volume for now
        velocity = vol_data[:, :3]
        
        return geo_mesh, surf_mesh, surf_data, vol_mesh, velocity
