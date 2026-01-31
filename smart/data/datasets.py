from data.shapenetcar_dataset import ShapeNetCarDataset
from data.ahmedml_dataset import AhmedMLDataset
from data.shiftsuv_dataset import ShiftSUVDataset
from data.shiftwing_dataset import ShiftWingDataset


# Mapping of dataset names to their corresponding classes and properties
datasets = {"ShapeNetCar": {"dataset": ShapeNetCarDataset, "spatial_dim": 3, "surf_channels": 1, "vol_channels": 3, "params_dim": 0, "fields": {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}},
            "AhmedML": {"dataset": AhmedMLDataset, "spatial_dim": 3, "surf_channels": 1, "vol_channels": 3, "params_dim": 0, "fields": {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}},
            "ShiftSUV": {"dataset": ShiftSUVDataset, "spatial_dim": 3, "surf_channels": 1, "vol_channels": 3, "params_dim": 0, "fields": {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}},
            "ShiftWing": {"dataset": ShiftWingDataset, "spatial_dim": 3, "surf_channels": 1, "vol_channels": 3, "params_dim": 2, "fields": {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}}
           }


def get_dataset(config):
    """Returns the dataset based on the provided configuration.

    Args:
        config: Configuration object containing dataset parameters.

    Returns:
        tuple: A tuple containing:
            - train_data: Training dataset.
            - test_data: Testing dataset.
            - stats: Stats for normalization.
            - spatial_dim: Spatial dimension of the dataset.
            - surf_channels: Number of surface channels.
            - vol_channels: Number of volume channels.
            - params_dim: Number of dimensions of simulation parameters.
    """
    
    dataset = config.dataset
    data_path = config.data_path
    print(f"Using dataset {dataset} stored at {data_path}")

    if dataset in datasets:
        spatial_dim = datasets[dataset]["spatial_dim"]
        surf_channels = datasets[dataset]["surf_channels"]
        vol_channels = datasets[dataset]["vol_channels"]
        params_dim = datasets[dataset]["params_dim"]
        fields = datasets[dataset]["fields"]
        train_data = datasets[dataset]["dataset"](data_path,
                                                  if_test=False,
                                                  geometry_points=config.num_body_points,
                                                  surface_points=config.num_surface_points,
                                                  volume_points=config.num_volume_points,
                                                  scale_positions=config.scale_positions)
        test_data = datasets[dataset]["dataset"](data_path,
                                                 if_test=True,
                                                 geometry_points=config.num_body_points,
                                                 surface_points=config.num_surface_points,
                                                 volume_points=config.num_volume_points,
                                                 scale_positions=config.scale_positions)
        stats = [train_data.mean_surf_data, train_data.std_surf_data,
                train_data.mean_vol_data, train_data.std_vol_data]
    else:
        raise ValueError(f"Unknown dataset ({config.dataset}) which is not supported!")
    
    return train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields


def prepare_dataset(config):
    """Prepare the dataset based on the provided configuration. Preparation means storing each sample in a 
    numpy array to speed up data loading during training and computing mean and std for normalization.

    Args:
        config: Configuration object containing dataset parameters.
    """
    
    dataset = config.dataset
    data_path = config.data_path
    print(f"Preparing dataset {dataset} stored at {data_path}")

    if dataset in datasets:
        train_data = datasets[dataset]["dataset"](data_path, if_test=False, prepare_data=True, copy_to_node=False)
        print(f"Dataset length: {len(train_data)}")
    else:
        raise ValueError(f"Unknown dataset ({config.dataset}) which is not supported!")
    