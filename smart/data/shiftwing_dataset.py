import os
from pathlib import Path
import shutil
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


class ShiftWingDataset(Dataset):
    """Dataset for SHIFT-Wing data from Luminary Cloud (https://huggingface.co/datasets/luminary-shift/WING)
    
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
            self.min_pos = torch.tensor([-1000.0, -1000.0, -1000.0])
            self.max_pos = torch.tensor([1000.0, 1000.0, 1000.0])
        else:
            self.min_pos = torch.tensor([-1000.0, -1000.0, -1000.0])
            self.max_pos = torch.tensor([1000.0, 1000.0, 1000.0])
        
        self.geometry_points = geometry_points
        self.surface_points = surface_points
        self.volume_points = volume_points
        self.fast_approx_sampling = fast_approx_sampling
        self.file_path = os.path.abspath(saved_folder)
        
        # load all ids from the folders, we use only the first 1000 samples which is enough
        ids = [int(f.name.split('_')[1]) for f in os.scandir(self.file_path) if f.is_dir() and any(os.scandir(f.path)) and f.name[:6] == "sample"]
        ids.sort()
        ids = np.array(ids)[:1000]
        self.all_ids = ids
        
        # test and train split
        self.training_ids = [215,406,544,486,69,1505,584,1367,1374,1488,319,141,517,580,403,508,570,136,1554,283,1256,
                             1537,1525,1438,139,520,63,384,81,370,1403,1565,1426,1502,297,58,470,322,348,1477,145,1401,
                             101,545,1316,31,239,87,496,1421,1297,29,332,1416,345,408,328,411,1637,1534,513,1348,198,
                             1358,1439,1441,1487,1454,294,502,89,489,213,1635,1355,1201,1204,1526,1317,15,1220,182,1508,
                             73,1501,1545,244,1286,448,1393,132,1394,40,156,484,84,1628,14,1282,1498,499,303,1406,1541,
                             505,1449,323,1314,1507,1422,1585,1491,1360,82,467,279,6,382,381,196,361,1377,383,110,364,
                             1519,152,233,1582,1268,354,119,1203,1592,183,33,148,305,420,1594,327,1450,32,1346,77,292,
                             1216,390,1546,72,60,1458,400,1606,117,200,1615,125,1499,112,1327,397,553,11,378,92,1530,
                             395,1599,592,1259,1290,171,1611,1318,563,290,1424,1428,555,339,44,1571,1460,235,445,1533,
                             120,418,42,1550,1621,476,1378,22,566,1622,1365,1391,542,366,49,21,25,23,70,321,440,300,600,
                             306,272,1551,387,1543,1643,1601,315,130,571,1588,286,1225,1254,298,314,444,1476,5,1455,246,
                             1356,1417,1407,341,468,20,1445,1233,1610,1209,1224,251,389,1581,54,433,490,393,352,498,100,
                             74,1420,525,1575,1549,186,278,134,428,593,521,114,1383,142,1456,177,1528,1269,1518,1448,
                             1205,507,441,1305,1412,45,1230,316,1437,1583,365,1632,491,261,1239,1495,1638,1351,1462,
                             1580,506,419,362,228,207,1509,380,90,1279,2,1618,1207,83,1313,1212,1612,1608,1283,8,1246,
                             1362,1600,1517,43,449,504,1208,86,1222,537,36,121,557,457,309,560,432,573,1634,19,438,137,
                             1425,1529,564,116,78,1280,52,1636,65,1325,1237,469,413,75,558,431,1301,346,144,147,1385,
                             336,34,252,510,1553,299,1446,401,1552,1597,550,1573,1515,1430,547,586,551,515,143,30,1366,
                             1381,1398,178,313,211,1345,1287,1322,357,1215,1221,376,497,1251,568,1561,1520,574,1630,
                             1473,179,455,1296,1596,358,285,377,273,353,1329,575,1339,466,4,1595,1388,461,238,280,511,
                             1364,569,1629,1470,1375,404,91,151,587,1540,1244,212,514,1275,1341,1436,1324,582,126,
                             1404,356,67,210,1343,1578,93,265,581,1576,1371,527,230,226,1370,1334,1642,1633,556,1294,
                             1266,295,388,1263,76,396,39,97,1308,1480,439,153,1570,1299,482,180,1236,1617,220,559,293,
                             1380,1496,260,165,1466,349,483,1485,1368,1291,1447,55,1427,1218,255,1591,1336,549,289,59,
                             1513,131,1616,187,1567,363,331,1273,1620,562,541,1202,1411,1240,264,167,409,1579,1342,369,
                             311,1312,123,1484,599,95,478,355,64,1587,1489,394,104,1248,405,1400,53,257,118,1451,253,
                             159,1486,334,1527,317,1226,1369,1271,1413,1423,1210,423,111,481,1399,350,284,1309,589,188,
                             68,463,1217,535,480,1584,1547,1274,98,435,1333,1563,1229,195,1384,1261,1235,1337,494,1614,
                             1619,1555,351,1410,107,325,258,436,1598,1387,1471,368,1250,1475,338,1243,1277,529,268,1408,
                             452,1435,1335,24,434,479,534,1493,1532,578,1465,442,451,61,1350,1604,164,225,1464,1548,199,
                             1483,1319,62,27,516,194,1523,240,1559,1363,1349,1442,540,96,79,277,28,1521,1264,443,1627,
                             219,122,1278,1315,410,528,1245,17,37,209,1415,191,1262,1624,1557,1219,412,41,46,1605,421,
                             254,231,524,312,342,206,447,222,56,501,1,492,454,1607,392,85,347,1593,189,135,1481,3,320,
                             561,379,546,1625,203,1474,465,531,576,450,214,430,1361,1457,1376,150,146,310,232,1213,1490,
                             1431,1640,1562,597,166,275,249,456,1258,1568,1443,595,1340,536,168,532,1276,567,302,263,
                             1310,485,12,259,236,1252,458,1397,591,262,245,1293,1395,227,360,1512,1641,1558,1206,1535,
                             35,372,172,1409,103,1227,1482,274,1590,426,1382,1390,464,281,1418,548,192,522,473,9,375,
                             256,18,343,523,1603,330,1247,158,115,218,234,26,13,500,386,162,1511,554,1228,1452,276,
                             1354,1389,1602,224,326,519,269,1311,51,47,337,399,1321,1332]
        self.test_ids = [1500,288,488,1284,243,1467,1572,108,538,598,155,296,1504,1524,242,1586,385,1292,127,129,1353,
                         10,474,1347,446,402,493,1211,157,1613,197,324,487,113,416,1402,308,1574,367,237,543,552,437,
                         1463,1242,1589,1295,204,1492,1214,109,7,1281,407,1494,1260,1461,1433,1544,163,175,301,1503,
                         565,229,38,1285,1288,1468,1645,1419,1440,462,201,1328,1241,1323,208,307,217,170,359,427,1556,
                         1307,1538,391,1396,241,71,1560,340,99,154,588,1536,190,1302,247,415,471,138,161,1234,124,1444,
                         181,371,424,173,223,16,1379,1306,1232,1434,1255,530,429,1267,1478,128,1479,1459,577,495,184,
                         425,1531,594,526,149,1506,1631,1326,1372,1644,1320,373,57,248,422,1300,533,503,329,66,417,176,
                         539,216,1249,267,1609,1330,1639,1510,88,133,1338,1539,1352,1238,1414,221,140,1272,453,590,287,
                         414,374,572,1253,1497,169,1373,585,1357,106,1623,1331,174,472,1392,1257,304,270,1453,1270,1429,
                         583,1522,250,335,1386,1303,1577,202,1265]
        
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
            self.copy_data_to_node(f"/data/scratch/{user}/data/shift-wing")
        
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
                src_folder = os.path.join(self.file_path, f"sample_{id:06d}")
                dst_folder = os.path.join(path, f"sample_{id:06d}")
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder, exist_ok=True)
                for file in os.listdir(src_folder):
                    if file.endswith(".npy") or file.endswith(".json"):
                        src_file = os.path.join(src_folder, file)
                        dst_file = os.path.join(dst_folder, file)
                        if not os.path.exists(dst_file):
                            shutil.copy(src_file, dst_file)
            
            # Copy stats files if they exist
            stats_files = ["volume_stats.npy", "surface_stats.npy", "position_stats.npy", "params_stats.npy"]
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
            folder = os.path.join(self.file_path, f"sample_{id:06d}")
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
        params_stats_file = Path(os.path.join(self.file_path, "params_stats.npy"))

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
            
            # Parameters
            data = np.load(params_stats_file)
            self.min_params = torch.tensor(data[0])
            self.max_params = torch.tensor(data[1])
            
            print(f"Average surface: {self.mean_surf_data}")
            print(f"Average volume: {self.mean_vol_data}")
            print(f"Std surface: {self.std_surf_data}")
            print(f"Std volume: {self.std_vol_data}")
            print(f"Min position: {self.min_pos}")
            print(f"Max position: {self.max_pos}")
            print(f"Min parameters: {self.min_params}")
            print(f"Max parameters: {self.max_params}")
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

        min_params = torch.full((2,), np.inf, dtype=torch.float32)
        max_params = torch.full((2,), -np.inf, dtype=torch.float32)

        # Iterate over training samples
        for id in self.training_ids:
            print(f"Computing stats for sample {id}")
            folder = os.path.join(self.file_path, f"sample_{id:06d}")
            mesh = self.get_surface_mesh(folder, id)
            surf_mesh, surf_data = self.get_surface_data(folder, id)
            vol_mesh, vol_data = self.get_volume_data(folder, id)
            params = self.get_angle_mach(folder, id)
            
            for d in range(3):
                max_pos[d] = max(max_pos[d], mesh[:, d].max().item(), surf_mesh[:, d].max().item(), vol_mesh[:, d].max().item())
                min_pos[d] = min(min_pos[d], mesh[:, d].min().item(), surf_mesh[:, d].min().item(), vol_mesh[:, d].min().item())
            
            surf_data_sum += surf_data.sum(dim=0)
            vol_data_sum += vol_data.sum(dim=0)

            surf_data_squared_sum += (surf_data ** 2).sum(dim=0)
            vol_data_squared_sum += (vol_data ** 2).sum(dim=0)

            surf_data_count += surf_data.shape[0]
            vol_data_count += vol_data.shape[0]

            for d in range(2):
                min_params[d] = min(min_params[d], params[d].item())
                max_params[d] = max(max_params[d], params[d].item())

        std = lambda sum, squared_sum, count: torch.sqrt((squared_sum - ((sum ** 2) / count)) / (count-1))
        
        self.mean_surf_data = surf_data_sum / surf_data_count
        self.std_surf_data = std(surf_data_sum, surf_data_squared_sum, surf_data_count)
        
        self.mean_vol_data = vol_data_sum / vol_data_count
        self.std_vol_data = std(vol_data_sum, vol_data_squared_sum, vol_data_count)
        
        self.min_pos = min_pos
        self.max_pos = max_pos
        
        self.min_params = min_params
        self.max_params = max_params

        # Save the stats to a file for future use
        vol_stats_file = Path(os.path.join(self.file_path, "volume_stats.npy"))
        surf_stats_file = Path(os.path.join(self.file_path, "surface_stats.npy"))
        pos_stats_file = Path(os.path.join(self.file_path, "position_stats.npy"))
        params_stats_file = Path(os.path.join(self.file_path, "params_stats.npy"))
        np.save(surf_stats_file, np.array([self.mean_surf_data, self.std_surf_data]))
        np.save(vol_stats_file, np.array([self.mean_vol_data, self.std_vol_data]))
        np.save(pos_stats_file, np.array([self.min_pos, self.max_pos]))
        np.save(params_stats_file, np.array([self.min_params, self.max_params]))

        print(f"Average surface: {self.mean_surf_data}")
        print(f"Average volume: {self.mean_vol_data}")
        print(f"Std surface: {self.std_surf_data}")
        print(f"Std volume: {self.std_vol_data}")
        print(f"Min position: {self.min_pos}")
        print(f"Max position: {self.max_pos}")
        print(f"Min params: {self.min_params}")
        print(f"Max params: {self.max_params}")

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
            data = torch.tensor(vtk_to_numpy(cell_data.GetArray("Pressure (Pa)")), dtype=torch.float32)
            #wall_shear_stress = torch.tensor(vtk_to_numpy(cell_data.GetArray("Wall Shear Stress (N/m²)")), dtype=torch.float32)
            
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
            velocity = torch.tensor(vtk_to_numpy(point_data.GetArray("Velocity (m/s)")), dtype=torch.float32)
            #pressure = torch.tensor(vtk_to_numpy(point_data.GetArray("Pressure (Pa)")), dtype=torch.float32)
            data = torch.tensor(velocity.numpy())

            # Save the mesh and velocity data to a numpy file for future use
            np.save(os.path.join(folder, "volume.npy"), mesh.numpy())
            np.save(os.path.join(folder, "volume_data.npy"), data)
        else:
            # Load the mesh and velocity data from the saved numpy files
            mesh = torch.tensor(np.load(os.path.join(folder, "volume.npy")), dtype=torch.float32)
            data = torch.tensor(np.load(os.path.join(folder, "volume_data.npy")), dtype=torch.float32)
        
        return mesh, data
    
    def get_angle_mach(self, folder, idx):
        with open(os.path.join(folder, f"params.json")) as f:
            params = json.load(f)
            params = torch.tensor([params['alpha'], params['mach']], dtype=torch.float32)
        
        return params
        
    
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
                - torch.Tensor: The parameters (angle of attack and Mach number) of the data sample as tensor with shape (number of parameters).
        """
        # lLoad the data for the given index
        folder = os.path.join(self.file_path, f"sample_{self.data[idx]:06d}")
        geo_mesh = self.get_surface_mesh(folder, idx)
        surf_mesh, surf_data = self.get_surface_data(folder, idx)
        vol_mesh, vol_data = self.get_volume_data(folder, idx)
        params = self.get_angle_mach(folder, idx)
        
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
        
        params = params / self.max_params
        
        return geo_mesh, surf_mesh, surf_data, vol_mesh, velocity, params
