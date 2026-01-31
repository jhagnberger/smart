import torch


class RelL2Loss():
    """Relative L2 loss for PDEs adopted from https://github.com/BaratiLab/FactFormer/blob/main/loss_fn.py"""

    def __init__(self, dim=-2, eps=1e-5, reduction='sum', reduce_all=True):
        self.dim = dim
        self.eps = eps
        self.reduction = reduction
        self.reduce_all = reduce_all
    
    def __call__(self, y_hat, y):
        assert y_hat.shape == y.shape

        reduce_fn = torch.mean if self.reduction == 'mean' else torch.sum

        y_norm = reduce_fn((y ** 2), dim=self.dim)
        mask = y_norm < self.eps
        y_norm[mask] = self.eps
        diff = reduce_fn((y_hat - y) ** 2, dim=self.dim)
        diff = diff / y_norm  # [b, c]
        
        if self.reduce_all:
            diff = diff.sqrt().mean() # mean across channels and batch and any other dimensions
        else:
            diff = diff.sqrt() # do nothing
        return diff


class CombinedLoss():
    """Computes a combined loss by summing over independent losses for the surface and volume fields.
    
    Args:
        loss_fn: Loss function to be used for both surface and volume fields.
        fields: Dictionary specifying the fields present in the dataset.
    """
    
    def __init__(self, loss_fn, fields):
        self.loss_fn = loss_fn
        self.fields = fields
    
    def __call__(self, y_hat_surf, y_hat_vol, y_surf, y_vol):
        """Computes the combined loss by summing over independent losses for surface and volume fields.
        
        Args:
            y_hat_surf: Predicted surface data tensor.
            y_hat_vol: Predicted volume data tensor.
            y_surf: Ground truth surface data tensor.
            y_vol: Ground truth volume data tensor.
            
        Returns:
            torch.Tensor: The combined loss value.
        """
        if self.fields["surface"] == ["pressure"]:
            loss_press = self.loss_fn(y_hat_surf, y_surf)
            loss_velo = self.loss_fn(y_hat_vol, y_vol)
            loss = loss_velo + loss_press
        elif self.fields["surface"] == ["pressure", "wall_shear_stress_x", "wall_shear_stress_y", "wall_shear_stress_z"]:
            loss_press = self.loss_fn(y_hat_surf[..., 0:1], y_surf[..., 0:1])
            loss_wss = self.loss_fn(y_hat_surf[..., 1:4], y_surf[..., 1:4])
            loss_velo = self.loss_fn(y_hat_vol[..., :], y_vol)
            loss = loss_velo + loss_press + loss_wss
        else:
            raise ValueError("Unsupported fields for loss computation.")
        
        return loss
