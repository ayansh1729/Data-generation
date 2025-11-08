"""
Data augmentation pipelines for diffusion models
"""

import torch
import torchvision.transforms as transforms
from typing import Optional, List, Tuple, Callable
import numpy as np


def get_augmentation_pipeline(
    image_size: int = 64,
    mode: str = 'train',
    config: Optional[dict] = None
) -> transforms.Compose:
    """
    Get data augmentation pipeline.
    
    Args:
        image_size: Target image size
        mode: 'train', 'val', or 'test'
        config: Optional configuration dictionary
        
    Returns:
        Composed transform pipeline
    """
    if config is None:
        config = {}
    
    if mode == 'train':
        transform_list = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        
        # Optional augmentations
        if config.get('horizontal_flip', True):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if config.get('vertical_flip', False):
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        
        if config.get('random_crop', False):
            transform_list.insert(1, transforms.RandomCrop(image_size, padding=4))
        
        if config.get('color_jitter', False):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            )
        
        if config.get('random_rotation', False):
            transform_list.append(transforms.RandomRotation(15))
        
        if config.get('random_affine', False):
            transform_list.append(
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                )
            )
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
    
    else:
        # Validation/Test: minimal augmentation
        transform_list = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    
    return transforms.Compose(transform_list)


class MixUp:
    """
    MixUp augmentation for diffusion models.
    Mixes two samples with a random ratio.
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Apply MixUp to a batch.
        
        Args:
            batch: Input batch (B, C, H, W)
            
        Returns:
            Mixed batch
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        mixed_batch = lam * batch + (1 - lam) * batch[index]
        
        return mixed_batch


class CutMix:
    """
    CutMix augmentation for diffusion models.
    Cuts and pastes patches between samples.
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Apply CutMix to a batch.
        
        Args:
            batch: Input batch (B, C, H, W)
            
        Returns:
            Cut-mixed batch
        """
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        # Get random bounding box
        _, _, H, W = batch.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        # Random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Bounding box
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        mixed_batch = batch.clone()
        mixed_batch[:, :, y1:y2, x1:x2] = batch[index, :, y1:y2, x1:x2]
        
        return mixed_batch


class RandomNoise:
    """Add random noise to images."""
    
    def __init__(self, noise_level: float = 0.1, noise_type: str = 'gaussian'):
        self.noise_level = noise_level
        self.noise_type = noise_type
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Add noise to image.
        
        Args:
            img: Input image
            
        Returns:
            Noisy image
        """
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(img) * self.noise_level
            return img + noise
        
        elif self.noise_type == 'uniform':
            noise = (torch.rand_like(img) - 0.5) * 2 * self.noise_level
            return img + noise
        
        else:
            return img


class RandomMask:
    """Randomly mask patches of the image."""
    
    def __init__(self, mask_ratio: float = 0.15, patch_size: int = 8):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply random masking.
        
        Args:
            img: Input image (C, H, W)
            
        Returns:
            Masked image
        """
        C, H, W = img.shape
        
        # Number of patches
        n_patches_h = H // self.patch_size
        n_patches_w = W // self.patch_size
        n_patches = n_patches_h * n_patches_w
        
        # Number of patches to mask
        n_mask = int(n_patches * self.mask_ratio)
        
        # Random mask indices
        mask_indices = np.random.choice(n_patches, n_mask, replace=False)
        
        # Apply mask
        masked_img = img.clone()
        for idx in mask_indices:
            i = idx // n_patches_w
            j = idx % n_patches_w
            
            y1 = i * self.patch_size
            y2 = (i + 1) * self.patch_size
            x1 = j * self.patch_size
            x2 = (j + 1) * self.patch_size
            
            masked_img[:, y1:y2, x1:x2] = 0
        
        return masked_img


class TabularAugmentation:
    """Augmentation for tabular data."""
    
    def __init__(
        self,
        noise_level: float = 0.05,
        dropout_prob: float = 0.1,
        swap_prob: float = 0.1
    ):
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob
        self.swap_prob = swap_prob
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to tabular data.
        
        Args:
            data: Input data (B, D) or (D,)
            
        Returns:
            Augmented data
        """
        augmented = data.clone()
        
        # Add noise
        if self.noise_level > 0:
            noise = torch.randn_like(augmented) * self.noise_level
            augmented = augmented + noise
        
        # Random dropout
        if self.dropout_prob > 0:
            mask = torch.rand_like(augmented) > self.dropout_prob
            augmented = augmented * mask
        
        # Random feature swapping
        if self.swap_prob > 0 and augmented.dim() == 2:
            batch_size, n_features = augmented.shape
            n_swaps = int(n_features * self.swap_prob)
            
            for _ in range(n_swaps):
                if n_features > 1:
                    i, j = np.random.choice(n_features, 2, replace=False)
                    augmented[:, [i, j]] = augmented[:, [j, i]]
        
        return augmented


class ComposedAugmentation:
    """Compose multiple augmentations."""
    
    def __init__(self, augmentations: List[Callable]):
        self.augmentations = augmentations
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply all augmentations sequentially.
        
        Args:
            data: Input data
            
        Returns:
            Augmented data
        """
        for aug in self.augmentations:
            data = aug(data)
        return data


def get_strong_augmentation(image_size: int = 64) -> transforms.Compose:
    """Get strong augmentation pipeline for training."""
    return transforms.Compose([
        transforms.Resize(image_size + 8),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


def get_weak_augmentation(image_size: int = 64) -> transforms.Compose:
    """Get weak augmentation pipeline."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentation pipelines...")
    
    # Test image augmentation
    transform = get_augmentation_pipeline(image_size=64, mode='train')
    from PIL import Image
    img = Image.new('RGB', (128, 128), color='red')
    augmented = transform(img)
    print(f"Augmented image shape: {augmented.shape}")
    print(f"Augmented image range: [{augmented.min():.2f}, {augmented.max():.2f}]")
    
    # Test MixUp
    batch = torch.randn(8, 3, 64, 64)
    mixup = MixUp(alpha=1.0)
    mixed = mixup(batch)
    print(f"MixUp output shape: {mixed.shape}")
    
    # Test CutMix
    cutmix = CutMix(alpha=1.0)
    cut_mixed = cutmix(batch)
    print(f"CutMix output shape: {cut_mixed.shape}")
    
    # Test tabular augmentation
    tabular_data = torch.randn(100, 10)
    tab_aug = TabularAugmentation(noise_level=0.05)
    augmented_tab = tab_aug(tabular_data)
    print(f"Augmented tabular shape: {augmented_tab.shape}")
    
    print("All tests passed!")

