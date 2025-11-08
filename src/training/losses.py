"""
Loss functions for diffusion models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


def simple_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Simple MSE loss (L2 loss).
    
    Args:
        pred: Predicted values
        target: Target values
        
    Returns:
        Loss value
    """
    return F.mse_loss(pred, target)


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 loss (Mean Absolute Error).
    
    Args:
        pred: Predicted values
        target: Target values
        
    Returns:
        Loss value
    """
    return F.l1_loss(pred, target)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Huber loss (smooth L1 loss).
    
    Args:
        pred: Predicted values
        target: Target values
        delta: Threshold for switching between L1 and L2
        
    Returns:
        Loss value
    """
    return F.smooth_l1_loss(pred, target, beta=delta)


def hybrid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float = 0.5,
    l2_weight: float = 0.5
) -> torch.Tensor:
    """
    Hybrid L1 + L2 loss.
    
    Args:
        pred: Predicted values
        target: Target values
        l1_weight: Weight for L1 loss
        l2_weight: Weight for L2 loss
        
    Returns:
        Weighted loss value
    """
    l1 = F.l1_loss(pred, target)
    l2 = F.mse_loss(pred, target)
    return l1_weight * l1 + l2_weight * l2


def perceptual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    feature_extractor: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    Perceptual loss using feature similarity.
    
    Args:
        pred: Predicted images
        target: Target images
        feature_extractor: Network for extracting features (e.g., VGG)
        
    Returns:
        Perceptual loss value
    """
    if feature_extractor is None:
        # Fallback to pixel-wise loss
        return F.mse_loss(pred, target)
    
    # Extract features
    pred_features = feature_extractor(pred)
    target_features = feature_extractor(target)
    
    # Compute feature-wise loss
    if isinstance(pred_features, (list, tuple)):
        loss = sum(F.mse_loss(p, t) for p, t in zip(pred_features, target_features))
        loss = loss / len(pred_features)
    else:
        loss = F.mse_loss(pred_features, target_features)
    
    return loss


class WeightedLoss(nn.Module):
    """
    Time-weighted loss for diffusion models.
    Weights loss differently at different timesteps.
    """
    
    def __init__(
        self,
        base_loss: Callable = F.mse_loss,
        weight_type: str = 'constant',
        **kwargs
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weight_type = weight_type
        self.kwargs = kwargs
    
    def get_weights(self, t: torch.Tensor, max_t: int = 1000) -> torch.Tensor:
        """
        Compute time-dependent weights.
        
        Args:
            t: Timesteps
            max_t: Maximum timestep
            
        Returns:
            Weights for each timestep
        """
        if self.weight_type == 'constant':
            return torch.ones_like(t, dtype=torch.float32)
        
        elif self.weight_type == 'sqrt':
            # More weight on early timesteps
            return torch.sqrt((max_t - t).float())
        
        elif self.weight_type == 'linear':
            # Linear weighting
            return (max_t - t).float() / max_t
        
        elif self.weight_type == 'exponential':
            # Exponential weighting
            return torch.exp(-t.float() / max_t)
        
        elif self.weight_type == 'snr':
            # Signal-to-noise ratio weighting
            alpha_bar = (1 - t.float() / max_t) ** 2
            return alpha_bar / (1 - alpha_bar + 1e-8)
        
        else:
            return torch.ones_like(t, dtype=torch.float32)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            pred: Predicted values
            target: Target values
            t: Timesteps
            
        Returns:
            Weighted loss
        """
        # Compute base loss
        if 'reduction' not in self.kwargs:
            loss = self.base_loss(pred, target, reduction='none')
        else:
            loss = self.base_loss(pred, target, **self.kwargs)
        
        # Apply time weights if provided
        if t is not None and self.weight_type != 'constant':
            weights = self.get_weights(t)
            weights = weights.view(-1, *([1] * (loss.ndim - 1)))
            loss = loss * weights
        
        return loss.mean()


class VLBLoss(nn.Module):
    """
    Variational Lower Bound (VLB) loss for diffusion models.
    Based on the ELBO objective.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VLB loss.
        
        Args:
            x_start: Clean images
            t: Timesteps
            noise: Noise tensor
            
        Returns:
            VLB loss
        """
        # This is a simplified version
        # Full implementation would compute KL divergence terms
        
        # Forward diffusion
        x_t = self.model.q_sample(x_start, t, noise)
        
        # Predict noise
        pred_noise = self.model.model(x_t, t)
        
        # Compute loss
        loss = F.mse_loss(pred_noise, noise)
        
        return loss


def get_loss_function(
    loss_type: str = 'mse',
    **kwargs
) -> Callable:
    """
    Get loss function by name.
    
    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments
        
    Returns:
        Loss function
    """
    if loss_type == 'mse' or loss_type == 'l2':
        return simple_loss
    
    elif loss_type == 'l1' or loss_type == 'mae':
        return l1_loss
    
    elif loss_type == 'huber' or loss_type == 'smooth_l1':
        delta = kwargs.get('delta', 1.0)
        return lambda pred, target: huber_loss(pred, target, delta)
    
    elif loss_type == 'hybrid':
        l1_weight = kwargs.get('l1_weight', 0.5)
        l2_weight = kwargs.get('l2_weight', 0.5)
        return lambda pred, target: hybrid_loss(pred, target, l1_weight, l2_weight)
    
    elif loss_type == 'weighted':
        base_loss = kwargs.get('base_loss', F.mse_loss)
        weight_type = kwargs.get('weight_type', 'constant')
        return WeightedLoss(base_loss, weight_type, **kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    pred = torch.randn(4, 3, 64, 64)
    target = torch.randn(4, 3, 64, 64)
    t = torch.randint(0, 1000, (4,))
    
    # Test simple losses
    print(f"MSE Loss: {simple_loss(pred, target):.4f}")
    print(f"L1 Loss: {l1_loss(pred, target):.4f}")
    print(f"Huber Loss: {huber_loss(pred, target):.4f}")
    print(f"Hybrid Loss: {hybrid_loss(pred, target):.4f}")
    
    # Test weighted loss
    weighted = WeightedLoss(weight_type='linear')
    print(f"Weighted Loss: {weighted(pred, target, t):.4f}")
    
    # Test getting loss by name
    loss_fn = get_loss_function('mse')
    print(f"Loss from factory: {loss_fn(pred, target):.4f}")
    
    print("All tests passed!")

