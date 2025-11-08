"""
Visualization utilities for diffusion models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Union
import torchvision.utils as vutils
from pathlib import Path


def plot_samples(
    samples: torch.Tensor,
    nrow: int = 8,
    title: str = "Generated Samples",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12),
    normalize: bool = True
) -> plt.Figure:
    """
    Plot a grid of generated samples.
    
    Args:
        samples: Tensor of samples (B, C, H, W)
        nrow: Number of images per row
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        normalize: Whether to normalize images
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Make grid
    grid = vutils.make_grid(samples, nrow=nrow, padding=2, normalize=normalize)
    
    # Convert to numpy and transpose
    grid_np = grid.cpu().numpy()
    grid_np = np.transpose(grid_np, (1, 2, 0))
    
    # Plot
    plt.imshow(grid_np)
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        losses: List of training losses
        val_losses: Optional list of validation losses
        title: Plot title
        save_path: Optional path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(losses, label='Training Loss', linewidth=2)
    
    if val_losses:
        ax.plot(val_losses, label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_diffusion_trajectory(
    trajectory: torch.Tensor,
    timesteps: List[int],
    sample_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 3)
) -> plt.Figure:
    """
    Plot the diffusion process trajectory.
    
    Args:
        trajectory: Trajectory tensor (steps, batch, C, H, W)
        timesteps: List of timestep values
        sample_idx: Which sample to visualize
        save_path: Optional path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    num_steps = len(trajectory)
    fig, axes = plt.subplots(1, num_steps, figsize=figsize)
    
    if num_steps == 1:
        axes = [axes]
    
    for i, (img, t) in enumerate(zip(trajectory, timesteps)):
        # Get image
        img_np = img[sample_idx].cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        
        # Normalize
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Handle grayscale
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
        
        # Plot
        axes[i].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
        axes[i].set_title(f't={t}', fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_comparison(
    real_samples: torch.Tensor,
    generated_samples: torch.Tensor,
    nrow: int = 8,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot comparison between real and generated samples.
    
    Args:
        real_samples: Real data samples
        generated_samples: Generated samples
        nrow: Number of images per row
        save_path: Optional path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Real samples
    real_grid = vutils.make_grid(real_samples[:64], nrow=nrow, normalize=True)
    real_grid = real_grid.cpu().numpy().transpose(1, 2, 0)
    axes[0].imshow(real_grid)
    axes[0].set_title('Real Samples', fontsize=14)
    axes[0].axis('off')
    
    # Generated samples
    gen_grid = vutils.make_grid(generated_samples[:64], nrow=nrow, normalize=True)
    gen_grid = gen_grid.cpu().numpy().transpose(1, 2, 0)
    axes[1].imshow(gen_grid)
    axes[1].set_title('Generated Samples', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_metrics_dashboard(
    metrics: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot a dashboard of multiple metrics.
    
    Args:
        metrics: Dictionary of metric lists
        save_path: Optional path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, (name, values) in enumerate(metrics.items()):
        if idx < len(axes):
            axes[idx].plot(values, linewidth=2)
            axes[idx].set_title(name, fontsize=12)
            axes[idx].set_xlabel('Step', fontsize=10)
            axes[idx].set_ylabel('Value', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_samples_grid(
    samples: torch.Tensor,
    save_dir: Union[str, Path],
    epoch: int,
    nrow: int = 8
):
    """
    Save samples as a grid image.
    
    Args:
        samples: Generated samples
        save_dir: Directory to save
        epoch: Current epoch number
        nrow: Number of images per row
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    grid = vutils.make_grid(samples, nrow=nrow, normalize=True)
    vutils.save_image(grid, save_dir / f'samples_epoch_{epoch:04d}.png')


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Test plot_samples
    samples = torch.randn(64, 3, 32, 32)
    fig = plot_samples(samples, nrow=8)
    print("Sample grid created")
    plt.close(fig)
    
    # Test training curves
    losses = [10 - i * 0.1 + np.random.randn() * 0.1 for i in range(100)]
    val_losses = [10 - i * 0.08 + np.random.randn() * 0.15 for i in range(100)]
    fig = plot_training_curves(losses, val_losses)
    print("Training curves plotted")
    plt.close(fig)
    
    print("All tests passed!")

