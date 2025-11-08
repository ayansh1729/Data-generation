"""
GradCAM (Gradient-weighted Class Activation Mapping) for Diffusion Models

Implements GradCAM to visualize which parts of the input the model focuses on
during the denoising process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class GradCAMExplainer:
    """
    GradCAM explainer for diffusion models.
    
    Args:
        model: Diffusion model to explain
        target_layers: List of layer names to compute GradCAM for
    """
    
    def __init__(self, model: nn.Module, target_layers: Optional[List[str]] = None):
        self.model = model
        self.target_layers = target_layers or ["mid_attn"]
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for target layers."""
        
        def forward_hook(name):
            def hook(module, input, output):
                # Handle case where output is a tuple (e.g., from DownBlock)
                if isinstance(output, tuple):
                    # Save the first element (main output)
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                # grad_output is the gradient w.r.t. output of this layer
                # It's a tuple, and we want the first element if it exists
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_layers):
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_full_backward_hook(backward_hook(name)))
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_gradcam(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        layer_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GradCAM for the given input.
        
        Args:
            x: Input tensor
            t: Timestep tensor
            layer_name: Specific layer to compute GradCAM for (or None for all)
            
        Returns:
            Dictionary of GradCAM heatmaps for each layer
        """
        self.model.eval()
        x.requires_grad = True
        
        # Forward pass
        if hasattr(self.model, 'model'):
            output = self.model.model(x, t)
        else:
            output = self.model(x, t)
        
        # Backward pass
        self.model.zero_grad()
        target = output.mean()
        target.backward()
        
        # Compute GradCAM for each layer
        gradcams = {}
        
        for name in self.activations.keys():
            if layer_name is not None and layer_name not in name:
                continue
            
            if name not in self.gradients:
                continue
            
            # Get gradients and activations
            gradients = self.gradients[name]
            activations = self.activations[name]
            
            # Check if both have spatial dimensions (B, C, H, W)
            if gradients.dim() != 4 or activations.dim() != 4:
                # Skip this layer if it doesn't have spatial dimensions
                continue
            
            # Ensure spatial dimensions match
            if gradients.shape[2:] != activations.shape[2:]:
                continue
            
            # Global average pooling of gradients over spatial dimensions
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            
            # Weighted combination of activation maps
            cam = (weights * activations).sum(dim=1, keepdim=True)
            
            # ReLU
            cam = F.relu(cam)
            
            # Normalize per sample
            batch_size = cam.shape[0]
            for i in range(batch_size):
                cam_i = cam[i]
                cam_i = cam_i - cam_i.min()
                cam_i = cam_i / (cam_i.max() + 1e-8)
                cam[i] = cam_i
            
            # Resize to input size
            cam = F.interpolate(
                cam,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            gradcams[name] = cam.detach()
        
        return gradcams
    
    def visualize_gradcam(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        layer_name: Optional[str] = None,
        alpha: float = 0.5,
        colormap: str = "jet"
    ) -> Dict[str, np.ndarray]:
        """
        Visualize GradCAM overlaid on the input image.
        
        Args:
            x: Input tensor (B, C, H, W)
            t: Timestep tensor
            layer_name: Specific layer to visualize
            alpha: Overlay transparency
            colormap: Matplotlib colormap name
            
        Returns:
            Dictionary of visualization arrays
        """
        gradcams = self.compute_gradcam(x, t, layer_name)
        
        # Normalize input to [0, 1]
        x_np = x.detach().cpu().numpy()
        x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
        
        visualizations = {}
        
        for name, cam in gradcams.items():
            cam_np = cam.squeeze().cpu().numpy()
            
            # Apply colormap
            cmap = plt.get_cmap(colormap)
            
            batch_viz = []
            for i in range(x_np.shape[0]):
                # Get image
                img = np.transpose(x_np[i], (1, 2, 0))
                
                # Get heatmap
                heatmap = cmap(cam_np[i])[:, :, :3]
                
                # Overlay
                overlay = alpha * heatmap + (1 - alpha) * img
                overlay = np.clip(overlay, 0, 1)
                
                batch_viz.append(overlay)
            
            visualizations[name] = np.stack(batch_viz)
        
        return visualizations
    
    def explain_generation(
        self,
        x_start: torch.Tensor,
        timesteps: List[int] = [0, 250, 500, 750, 999],
        save_path: Optional[str] = None
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Explain the generation process at multiple timesteps.
        
        Args:
            x_start: Clean images to add noise to
            timesteps: List of timesteps to analyze
            save_path: Optional path to save visualizations
            
        Returns:
            Dictionary mapping timesteps to GradCAM results
        """
        device = x_start.device
        results = {}
        
        for t_val in timesteps:
            t = torch.full((x_start.shape[0],), t_val, device=device, dtype=torch.long)
            
            # Add noise
            if hasattr(self.model, 'q_sample'):
                x_t = self.model.q_sample(x_start, t)
            else:
                # Simple noise addition if q_sample not available
                noise = torch.randn_like(x_start)
                alpha = 1 - t_val / 1000
                x_t = alpha * x_start + (1 - alpha) * noise
            
            # Compute GradCAM
            gradcams = self.compute_gradcam(x_t, t)
            visualizations = self.visualize_gradcam(x_t, t)
            
            results[t_val] = {
                "gradcams": gradcams,
                "visualizations": visualizations,
                "noisy_images": x_t.detach()
            }
        
        if save_path:
            self._save_visualizations(results, save_path)
        
        return results
    
    def _save_visualizations(self, results: Dict, save_path: str):
        """Save visualizations to disk."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        for t_val, result in results.items():
            for layer_name, viz in result["visualizations"].items():
                # Save each image in the batch
                for i, img in enumerate(viz):
                    safe_layer_name = layer_name.replace(".", "_")
                    filename = f"gradcam_t{t_val}_layer_{safe_layer_name}_sample{i}.png"
                    filepath = os.path.join(save_path, filename)
                    
                    plt.figure(figsize=(6, 6))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"GradCAM at t={t_val}\nLayer: {layer_name}")
                    plt.tight_layout()
                    plt.savefig(filepath, dpi=150, bbox_inches='tight')
                    plt.close()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


def plot_gradcam_grid(
    gradcams: Dict[str, torch.Tensor],
    images: torch.Tensor,
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot a grid of GradCAM visualizations.
    
    Args:
        gradcams: Dictionary of GradCAM heatmaps
        images: Original images
        titles: Optional titles for each subplot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_layers = len(gradcams)
    fig, axes = plt.subplots(1, n_layers + 1, figsize=figsize)
    
    if n_layers == 0:
        return fig
    
    # Plot original image
    img = images[0].cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Plot GradCAM for each layer
    for idx, (name, cam) in enumerate(gradcams.items(), 1):
        cam_np = cam[0, 0].cpu().numpy()
        axes[idx].imshow(cam_np, cmap='jet')
        title = titles[idx-1] if titles and idx-1 < len(titles) else name
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test GradCAM
    from src.models.diffusion import DDPM
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = DDPM(image_size=64, channels=3, timesteps=1000).to(device)
    
    # Create explainer
    explainer = GradCAMExplainer(model, target_layers=["mid_attn", "mid_block1"])
    
    # Test on random input
    x = torch.randn(2, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    
    # Compute GradCAM
    gradcams = explainer.compute_gradcam(x, t)
    print(f"Computed GradCAM for layers: {list(gradcams.keys())}")
    
    # Visualize
    visualizations = explainer.visualize_gradcam(x, t)
    print(f"Generated visualizations with shape: {list(visualizations.values())[0].shape}")
    
    # Cleanup
    explainer.remove_hooks()

