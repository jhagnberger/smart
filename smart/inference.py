import hydra
from omegaconf import DictConfig

import torch
from timeit import default_timer

# Dataset and loss functions
from data.datasets import get_dataset
from utils.utils import initialize_gpu, get_model_checkpoint_name, count_model_params, get_optimizer_scheduler_loss, store_inference_results
from loss.losses import CombinedLoss

# SMART Model
from models.smart.smart import SMART


@hydra.main(version_base="1.2", config_path="config", config_name="car")
def main(cfg: DictConfig):
    # Extract config
    config = cfg.experiment
    
    # Overwrite the volume and surface query size to full dataset size during inference
    config.num_surface_points = -1
    config.num_volume_points = -1
    # Inference batch size of 1 to avoid dealing with batches of different sizes
    config.batch_size = 1
    
    # Set seed and GPU settings
    device = initialize_gpu(config.random_seed, high_precision=False)
    
    # Set gradient norm clipping, precision and amp
    gradient_norm = config.gradient_norm
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(config.precision, torch.float16)
    amp = config.amp
    print(gradient_norm, amp, dtype)

    # Load data
    train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               shuffle=True,
                                               prefetch_factor=56)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=config.batch_size,
                                              num_workers=config.num_workers,
                                              shuffle=False,
                                              prefetch_factor=56)
    # Move stats to device
    mean_surf = stats[0].to(device)
    std_surf = stats[1].to(device)
    mean_vol = stats[2].to(device)
    std_vol = stats[3].to(device)
    
    # Extract one training sample for inspection
    if params_dim > 0:
        sample_geo_mesh, sample_surf_mesh, sample_pressure, sample_vol_mesh, sample_velocity, params = train_data[0]
        print("Sample geo_mesh shape:", sample_geo_mesh.shape)
        print("Sample surf_mesh shape:", sample_surf_mesh.shape)
        print("Sample pressure shape:", sample_pressure.shape)
        print("Sample vol_mesh shape:", sample_vol_mesh.shape)
        print("Sample velocity shape:", sample_velocity.shape)
        print("Sample params shape:", params.shape)
    else:
        sample_geo_mesh, sample_surf_mesh, sample_pressure, sample_vol_mesh, sample_velocity = train_data[0]
        print("Sample geo_mesh shape:", sample_geo_mesh.shape)
        print("Sample surf_mesh shape:", sample_surf_mesh.shape)
        print("Sample pressure shape:", sample_pressure.shape)
        print("Sample vol_mesh shape:", sample_vol_mesh.shape)
        print("Sample velocity shape:", sample_velocity.shape)
    
    # Create model
    models = {"SMART": (SMART, {"surface_channels": surf_channels, "volume_channels": vol_channels, "parameter_channels": params_dim})}
    
    if config.model_name in models:
        merged_kwargs = {**models[config.model_name][1], **config.architecture} if "architecture" in config else models[config.model_name][1]
        print(f"Model kwargs: {merged_kwargs}")
        model = models[config.model_name][0](**merged_kwargs).to(device)
    else:
        raise ValueError("Unknown model class name!")
    model = model.to(device)

    print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    print(f"Checkpoint name: {model_checkpoint_name}")
    
    # Load checkpoint
    checkpoint = torch.load("checkpoints/" + model_checkpoint_name + "_last.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Training and evaluation
    _, _, loss_fn, rel_l2_loss_fn = get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=1)
    combined_loss_fn = CombinedLoss(loss_fn, fields)
    
    # Losses
    test_losses = {"loss": 0, "rel_l2": 0,
                    "rel_l2_surf": 0,  "rel_l2_vol": 0,
                    "rel_l2_press": 0, "rel_l2_wall_shear_stress": 0,
                    "rel_l2_velo": 0, "rel_l2_velo_x": 0, "rel_l2_velo_y": 0, "rel_l2_velo_z": 0}

    # Evaluation
    model.eval()
    t1 = default_timer()
    with torch.no_grad():
        for batch in test_loader:
            # b, n, c
            if params_dim > 0:
                geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
                params = params.to(device)
            else:
                geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
                params = None
            
            # Move to device
            geo_mesh = geo_mesh.to(device)
            surf_mesh = surf_mesh.to(device)
            surf_data = surf_data.to(device)
            vol_mesh = vol_mesh.to(device)
            vol_data = vol_data.to(device)
            
            # Forward pass
            if amp:
                with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                    y_hat_surf, y_hat_vol = model.inference(geo_mesh, surf_mesh, vol_mesh, params)
            else:
                y_hat_surf, y_hat_vol = model.inference(geo_mesh, surf_mesh, vol_mesh, params)
            
            # Denormalize
            pred_surf = y_hat_surf[..., :] * std_surf + mean_surf
            gt_surf = surf_data * std_surf + mean_surf
            pred_vol = y_hat_vol[..., :] * std_vol + mean_vol
            gt_vol = vol_data * std_vol + mean_vol
        
            # Metrics
            batch_size = surf_data.size(0)
            
            # Combine loss
            test_losses["loss"] += combined_loss_fn(y_hat_surf, y_hat_vol, surf_data, vol_data).item() * batch_size
            test_losses["rel_l2_surf"] += rel_l2_loss_fn(y_hat_surf, surf_data).item() * batch_size
            test_losses["rel_l2_vol"] += rel_l2_loss_fn(y_hat_vol, vol_data).item() * batch_size
            test_losses["rel_l2"] += test_losses["rel_l2_surf"] + test_losses["rel_l2_vol"]
            
            # Surface
            if fields["surface"] == ["pressure"]:
                test_losses["rel_l2_press"] += rel_l2_loss_fn(pred_surf, gt_surf).item() * batch_size
            elif fields["surface"] == ["pressure", "wall_shear_stress_x", "wall_shear_stress_y", "wall_shear_stress_z"]:
                test_losses["rel_l2_press"] += rel_l2_loss_fn(pred_surf[..., 0:1], gt_surf[..., 0:1]).item() * batch_size
                test_losses["rel_l2_wall_shear_stress"] += rel_l2_loss_fn(pred_surf[..., 1:], gt_surf[..., 1:]).item() * batch_size
            else:
                raise ValueError("Unsupported fields for loss computation.")
                
            # Velocity
            test_losses["rel_l2_velo"] += rel_l2_loss_fn(pred_vol, gt_vol).item() * batch_size
            test_losses["rel_l2_velo_x"] += rel_l2_loss_fn(pred_vol[..., 0:1], gt_vol[..., 0:1]).item() * batch_size
            test_losses["rel_l2_velo_y"] += rel_l2_loss_fn(pred_vol[..., 1:2], gt_vol[..., 1:2]).item() * batch_size
            test_losses["rel_l2_velo_z"] += rel_l2_loss_fn(pred_vol[..., 2:3], gt_vol[..., 2:3]).item() * batch_size

        t2 = default_timer()
        print(f"Inference time: {t2 - t1} seconds")
        
        # Divide by total number of samples to get mean
        for loss_name in test_losses.keys():
            test_losses[loss_name] /= len(test_loader.dataset)
            
        print(f"Test Losses: {test_losses}")
            
        # Store
        store_inference_results("results", model_checkpoint_name, test_losses)


if __name__ == "__main__":
    main()
    print("Inference done.")
