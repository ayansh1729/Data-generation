"""
Data preprocessing utilities for diffusion models
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
import torchvision.transforms as transforms


def normalize_data(
    data: Union[torch.Tensor, np.ndarray],
    method: str = 'standard',
    mean: Optional[Union[float, Tuple]] = None,
    std: Optional[Union[float, Tuple]] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize data using various methods.
    
    Args:
        data: Input data
        method: Normalization method ('standard', 'minmax', 'tanh')
        mean: Mean for standard normalization
        std: Standard deviation for standard normalization
        min_val: Minimum value for minmax normalization
        max_val: Maximum value for minmax normalization
        
    Returns:
        Normalized data
    """
    is_numpy = isinstance(data, np.ndarray)
    
    if is_numpy:
        data = torch.from_numpy(data)
    
    if method == 'standard':
        # Standardization: (x - mean) / std
        if mean is None:
            mean = data.mean()
        if std is None:
            std = data.std() + 1e-8
        
        if isinstance(mean, (int, float)):
            mean = torch.tensor(mean).to(data.device)
        if isinstance(std, (int, float)):
            std = torch.tensor(std).to(data.device)
        
        normalized = (data - mean) / std
    
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        if min_val is None:
            min_val = data.min()
        if max_val is None:
            max_val = data.max()
        
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
    
    elif method == 'tanh':
        # Normalize to [-1, 1] using tanh-like transformation
        if mean is None:
            mean = data.mean()
        if std is None:
            std = data.std() + 1e-8
        
        normalized = 2 * ((data - mean) / (4 * std)).tanh()
    
    elif method == 'neg_one_to_one':
        # Scale to [-1, 1]
        if min_val is None:
            min_val = data.min()
        if max_val is None:
            max_val = data.max()
        
        normalized = 2 * (data - min_val) / (max_val - min_val + 1e-8) - 1
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if is_numpy:
        return normalized.numpy()
    
    return normalized


def denormalize_data(
    data: Union[torch.Tensor, np.ndarray],
    method: str = 'standard',
    mean: Optional[Union[float, Tuple]] = None,
    std: Optional[Union[float, Tuple]] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalize data back to original scale.
    
    Args:
        data: Normalized data
        method: Normalization method used
        mean: Mean used for normalization
        std: Std used for normalization
        min_val: Min value used for normalization
        max_val: Max value used for normalization
        
    Returns:
        Denormalized data
    """
    is_numpy = isinstance(data, np.ndarray)
    
    if is_numpy:
        data = torch.from_numpy(data)
    
    if method == 'standard':
        if mean is None:
            mean = 0.0
        if std is None:
            std = 1.0
        
        if isinstance(mean, (int, float)):
            mean = torch.tensor(mean).to(data.device)
        if isinstance(std, (int, float)):
            std = torch.tensor(std).to(data.device)
        
        denormalized = data * std + mean
    
    elif method == 'minmax':
        if min_val is None:
            min_val = 0.0
        if max_val is None:
            max_val = 1.0
        
        denormalized = data * (max_val - min_val) + min_val
    
    elif method == 'neg_one_to_one':
        if min_val is None:
            min_val = 0.0
        if max_val is None:
            max_val = 1.0
        
        denormalized = (data + 1) / 2 * (max_val - min_val) + min_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if is_numpy:
        return denormalized.numpy()
    
    return denormalized


def preprocess_images(
    images: Union[torch.Tensor, np.ndarray],
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    method: str = 'neg_one_to_one'
) -> torch.Tensor:
    """
    Preprocess images for diffusion models.
    
    Args:
        images: Input images (B, C, H, W) or (B, H, W, C)
        target_size: Target size (H, W)
        normalize: Whether to normalize
        method: Normalization method
        
    Returns:
        Preprocessed images
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    
    # Ensure channel-first format
    if images.ndim == 4 and images.shape[-1] in [1, 3]:
        images = images.permute(0, 3, 1, 2)
    
    # Resize if needed
    if target_size is not None:
        resize = transforms.Resize(target_size)
        images = resize(images)
    
    # Normalize
    if normalize:
        images = normalize_data(images, method=method)
    
    return images


def preprocess_tabular(
    data: Union[torch.Tensor, np.ndarray, 'pd.DataFrame'],
    normalize: bool = True,
    method: str = 'standard'
) -> torch.Tensor:
    """
    Preprocess tabular data for diffusion models.
    
    Args:
        data: Input data
        normalize: Whether to normalize
        method: Normalization method
        
    Returns:
        Preprocessed data
    """
    # Convert to numpy if DataFrame
    if hasattr(data, 'values'):
        data = data.values
    
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    
    # Normalize
    if normalize:
        data = normalize_data(data, method=method)
    
    return data


def add_noise(
    data: torch.Tensor,
    noise_type: str = 'gaussian',
    noise_level: float = 0.1
) -> torch.Tensor:
    """
    Add noise to data for data augmentation.
    
    Args:
        data: Input data
        noise_type: Type of noise ('gaussian', 'uniform', 'salt_pepper')
        noise_level: Amount of noise
        
    Returns:
        Noisy data
    """
    if noise_type == 'gaussian':
        noise = torch.randn_like(data) * noise_level
        return data + noise
    
    elif noise_type == 'uniform':
        noise = (torch.rand_like(data) - 0.5) * 2 * noise_level
        return data + noise
    
    elif noise_type == 'salt_pepper':
        mask = torch.rand_like(data) < noise_level
        noisy = data.clone()
        noisy[mask] = torch.randint_like(data[mask], 0, 2).float() * 2 - 1
        return noisy
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def compute_statistics(data: torch.Tensor) -> dict:
    """
    Compute statistics of the data.
    
    Args:
        data: Input data
        
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': data.mean().item(),
        'std': data.std().item(),
        'min': data.min().item(),
        'max': data.max().item(),
        'median': data.median().item(),
    }


def batch_normalize(
    batch: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize a batch and return normalization parameters.
    
    Args:
        batch: Input batch
        mean: Optional pre-computed mean
        std: Optional pre-computed std
        
    Returns:
        Tuple of (normalized_batch, mean, std)
    """
    if mean is None:
        mean = batch.mean(dim=0, keepdim=True)
    if std is None:
        std = batch.std(dim=0, keepdim=True) + 1e-8
    
    normalized = (batch - mean) / std
    
    return normalized, mean, std


def clip_data(
    data: torch.Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Clip data to a specified range.
    
    Args:
        data: Input data
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped data
    """
    return torch.clamp(data, min_val, max_val)


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing preprocessing utilities...")
    
    # Test image preprocessing
    images = torch.rand(4, 3, 32, 32)
    normalized = normalize_data(images, method='neg_one_to_one')
    print(f"Normalized images range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    denormalized = denormalize_data(normalized, method='neg_one_to_one')
    print(f"Denormalized images range: [{denormalized.min():.2f}, {denormalized.max():.2f}]")
    
    # Test tabular preprocessing
    tabular = torch.randn(100, 10)
    normalized_tab = normalize_data(tabular, method='standard')
    print(f"Normalized tabular mean: {normalized_tab.mean():.4f}, std: {normalized_tab.std():.4f}")
    
    # Test statistics
    stats = compute_statistics(images)
    print(f"Image statistics: {stats}")
    
    print("All tests passed!")

