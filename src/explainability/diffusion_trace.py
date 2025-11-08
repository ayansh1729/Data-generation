"""
Diffusion Trajectory Tracer

Tracks and visualizes the denoising trajectory through the diffusion process,
showing how the model progressively refines random noise into realistic samples.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from tqdm import tqdm


class DiffusionTracer:
    """
    Trace and visualize the diffusion process trajectory.
    
    Args:
        model: Diffusion model
        save_frequency: How often to save intermediate steps
    """
    
    def __init__(self, model: nn.Module, save_frequency: int = 50):
        self.model = model
        self.save_frequency = save_frequency
        self.trajectories = []
    
    @torch.no_grad()
    def trace_sampling(
        self,
        batch_size: int = 4,
        save_steps: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Trace the sampling process from noise to image.
        
        Args:
            batch_size: Number of samples to generate
            save_steps: Specific timesteps to save (or None for all)
            
        Returns:
            Dictionary containing trajectories and metadata
        """
        device = next(self.model.parameters()).device
        
        if not hasattr(self.model, 'timesteps'):
            raise ValueError("Model must have 'timesteps' attribute")
        
        timesteps = self.model.timesteps
        
        # Determine which steps to save
        if save_steps is None:
            save_steps = list(range(0, timesteps, self.save_frequency)) + [timesteps - 1]
        
        shape = (batch_size, self.model.channels, self.model.image_size, self.model.image_size)
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        trajectories = []
        saved_timesteps = []
        predictions = []
        
        # Iteratively denoise
        for t_idx in tqdm(reversed(range(0, timesteps)), desc="Tracing diffusion"):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            
            # Save current state
            if t_idx in save_steps:
                trajectories.append(x.cpu().clone())
                saved_timesteps.append(t_idx)
            
            # Predict and denoise
            if hasattr(self.model, 'model'):
                pred = self.model.model(x, t)
            else:
                pred = self.model(x, t)
            
            if t_idx in save_steps:
                predictions.append(pred.cpu().clone())
            
            # Sample next step
            if hasattr(self.model, 'p_sample'):
                x = self.model.p_sample(x, t)
            else:
                # Simple denoising step
                alpha = 1 - t_idx / timesteps
                x = alpha * x + (1 - alpha) * pred
        
        return {
            "trajectories": torch.stack(trajectories),  # (steps, batch, C, H, W)
            "predictions": torch.stack(predictions) if predictions else None,
            "timesteps": torch.tensor(saved_timesteps),
            "final_samples": x.cpu()
        }
    
    def trace_forward_backward(
        self,
        x_start: torch.Tensor,
        timesteps: List[int] = [0, 250, 500, 750, 999]
    ) -> Dict[str, torch.Tensor]:
        """
        Trace both forward (noise addition) and backward (denoising) processes.
        
        Args:
            x_start: Clean starting images
            timesteps: Timesteps to analyze
            
        Returns:
            Dictionary with forward and backward trajectories
        """
        device = x_start.device
        
        forward_states = []
        backward_states = []
        noise_predictions = []
        
        for t_val in timesteps:
            t = torch.full((x_start.shape[0],), t_val, device=device, dtype=torch.long)
            
            # Forward: add noise
            if hasattr(self.model, 'q_sample'):
                x_noisy = self.model.q_sample(x_start, t)
            else:
                noise = torch.randn_like(x_start)
                alpha = 1 - t_val / self.model.timesteps
                x_noisy = alpha * x_start + (1 - alpha) * noise
            
            forward_states.append(x_noisy.cpu().clone())
            
            # Backward: predict noise
            with torch.no_grad():
                if hasattr(self.model, 'model'):
                    pred = self.model.model(x_noisy, t)
                else:
                    pred = self.model(x_noisy, t)
                
                noise_predictions.append(pred.cpu().clone())
                
                # Attempt to denoise
                if hasattr(self.model, 'predict_start_from_noise'):
                    x_recon = self.model.predict_start_from_noise(x_noisy, t, pred)
                    backward_states.append(x_recon.cpu().clone())
                else:
                    backward_states.append(x_start.cpu().clone())
        
        return {
            "forward_states": torch.stack(forward_states),
            "backward_states": torch.stack(backward_states),
            "noise_predictions": torch.stack(noise_predictions),
            "timesteps": torch.tensor(timesteps),
            "original": x_start.cpu()
        }
    
    def visualize_trajectory(
        self,
        trajectory: torch.Tensor,
        timesteps: torch.Tensor,
        sample_idx: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 3)
    ) -> plt.Figure:
        """
        Visualize the sampling trajectory for a single sample.
        
        Args:
            trajectory: Trajectory tensor (steps, batch, C, H, W)
            timesteps: Timestep values for each step
            sample_idx: Which sample to visualize
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        num_steps = len(trajectory)
        fig, axes = plt.subplots(1, num_steps, figsize=figsize)
        
        if num_steps == 1:
            axes = [axes]
        
        for i, (img, t) in enumerate(zip(trajectory, timesteps)):
            # Get image and normalize
            img_np = img[sample_idx].numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # Handle grayscale
            if img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)
            
            axes[i].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
            axes[i].set_title(f't={t.item()}', fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_forward_backward(
        self,
        results: Dict[str, torch.Tensor],
        sample_idx: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 6)
    ) -> plt.Figure:
        """
        Visualize forward and backward processes side by side.
        
        Args:
            results: Results from trace_forward_backward
            sample_idx: Which sample to visualize
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        forward = results["forward_states"]
        backward = results["backward_states"]
        timesteps = results["timesteps"]
        
        num_steps = len(timesteps)
        fig, axes = plt.subplots(3, num_steps, figsize=figsize)
        
        for i, t in enumerate(timesteps):
            # Forward (noisy)
            fwd_img = forward[i, sample_idx].numpy()
            fwd_img = np.transpose(fwd_img, (1, 2, 0))
            fwd_img = (fwd_img - fwd_img.min()) / (fwd_img.max() - fwd_img.min() + 1e-8)
            
            # Backward (denoised)
            bwd_img = backward[i, sample_idx].numpy()
            bwd_img = np.transpose(bwd_img, (1, 2, 0))
            bwd_img = (bwd_img - bwd_img.min()) / (bwd_img.max() - bwd_img.min() + 1e-8)
            
            # Noise prediction
            pred_img = results["noise_predictions"][i, sample_idx].numpy()
            pred_img = np.transpose(pred_img, (1, 2, 0))
            pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
            
            # Plot
            axes[0, i].imshow(fwd_img)
            axes[0, i].set_title(f'Noisy (t={t.item()})', fontsize=9)
            axes[0, i].axis('off')
            
            axes[1, i].imshow(pred_img)
            axes[1, i].set_title('Prediction', fontsize=9)
            axes[1, i].axis('off')
            
            axes[2, i].imshow(bwd_img)
            axes[2, i].set_title('Denoised', fontsize=9)
            axes[2, i].axis('off')
        
        # Add row labels
        axes[0, 0].set_ylabel('Forward\n(Add Noise)', fontsize=10, rotation=0, ha='right', va='center')
        axes[1, 0].set_ylabel('Model\nPrediction', fontsize=10, rotation=0, ha='right', va='center')
        axes[2, 0].set_ylabel('Backward\n(Denoise)', fontsize=10, rotation=0, ha='right', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_animation(
        self,
        trajectory: torch.Tensor,
        timesteps: torch.Tensor,
        sample_idx: int = 0,
        save_path: Optional[str] = None,
        fps: int = 10
    ):
        """
        Create an animated GIF of the diffusion process.
        
        Args:
            trajectory: Trajectory tensor (steps, batch, C, H, W)
            timesteps: Timestep values
            sample_idx: Which sample to animate
            save_path: Path to save the GIF
            fps: Frames per second
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        def update(frame):
            ax.clear()
            
            img = trajectory[frame, sample_idx].numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            ax.set_title(f'Diffusion Process - t={timesteps[frame].item()}', fontsize=14)
            ax.axis('off')
        
        anim = FuncAnimation(fig, update, frames=len(trajectory), interval=1000//fps)
        
        if save_path:
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
        
        plt.close()
        return anim


if __name__ == "__main__":
    # Test Diffusion Tracer
    from src.models.diffusion import DDPM
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = DDPM(image_size=32, channels=3, timesteps=1000).to(device)
    
    # Create tracer
    tracer = DiffusionTracer(model, save_frequency=100)
    
    # Test forward-backward tracing
    x = torch.randn(2, 3, 32, 32).to(device)
    results = tracer.trace_forward_backward(x, timesteps=[0, 250, 500, 750, 999])
    
    print(f"Forward states shape: {results['forward_states'].shape}")
    print(f"Backward states shape: {results['backward_states'].shape}")
    
    # Visualize
    fig = tracer.visualize_forward_backward(results, sample_idx=0)
    print("Visualization complete")
    plt.close(fig)

