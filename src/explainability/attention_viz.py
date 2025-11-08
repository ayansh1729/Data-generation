"""
Attention Visualization for Diffusion Models

Visualizes attention maps from self-attention and cross-attention layers
to understand what the model focuses on during generation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AttentionVisualizer:
    """
    Visualize attention mechanisms in diffusion models.
    
    Args:
        model: Diffusion model with attention layers
        layer_names: Specific attention layer names to visualize
    """
    
    def __init__(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        self.model = model
        self.layer_names = layer_names
        self.attention_maps = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention maps."""
        
        def attention_hook(name):
            def hook(module, input, output):
                # Store attention maps if available
                if hasattr(module, 'attn_weights'):
                    self.attention_maps[name] = module.attn_weights.detach()
                elif isinstance(output, tuple) and len(output) > 1:
                    # Some attention modules return (output, attention_weights)
                    self.attention_maps[name] = output[1].detach()
            return hook
        
        for name, module in self.model.named_modules():
            # Look for attention layers
            if any(keyword in name.lower() for keyword in ['attn', 'attention']):
                if self.layer_names is None or any(ln in name for ln in self.layer_names):
                    self.hooks.append(module.register_forward_hook(attention_hook(name)))
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_attention_maps(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps from a forward pass.
        
        Args:
            x: Input tensor
            t: Timestep tensor
            
        Returns:
            Dictionary of attention maps
        """
        self.attention_maps = {}
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(self.model, 'model'):
                _ = self.model.model(x, t)
            else:
                _ = self.model(x, t)
        
        return self.attention_maps
    
    def visualize_attention_map(
        self,
        attention_map: torch.Tensor,
        title: str = "Attention Map",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Visualize a single attention map.
        
        Args:
            attention_map: Attention tensor (heads, tokens, tokens) or (tokens, tokens)
            title: Plot title
            save_path: Optional path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Handle different attention map shapes
        if attention_map.ndim == 4:
            # (batch, heads, tokens, tokens) -> take first batch, average heads
            attention_map = attention_map[0].mean(0)
        elif attention_map.ndim == 3:
            # (heads, tokens, tokens) -> average heads
            attention_map = attention_map.mean(0)
        
        attn_np = attention_map.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            attn_np,
            cmap='viridis',
            ax=ax,
            cbar=True,
            square=True,
            xticklabels=False,
            yticklabels=False
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Key Positions')
        ax.set_ylabel('Query Positions')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_all_attention_maps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        save_dir: Optional[str] = None,
        max_maps: int = 10
    ) -> Dict[str, plt.Figure]:
        """
        Visualize all attention maps in the model.
        
        Args:
            x: Input tensor
            t: Timestep tensor
            save_dir: Optional directory to save figures
            max_maps: Maximum number of maps to visualize
            
        Returns:
            Dictionary of figures
        """
        attention_maps = self.extract_attention_maps(x, t)
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        figures = {}
        
        for idx, (name, attn_map) in enumerate(list(attention_maps.items())[:max_maps]):
            save_path = None
            if save_dir:
                safe_name = name.replace(".", "_").replace("/", "_")
                save_path = Path(save_dir) / f"attention_{safe_name}.png"
            
            fig = self.visualize_attention_map(
                attn_map,
                title=f"Attention Map: {name}",
                save_path=str(save_path) if save_path else None
            )
            figures[name] = fig
            plt.close(fig)
        
        return figures
    
    def visualize_attention_heads(
        self,
        attention_map: torch.Tensor,
        num_heads: int = 8,
        figsize: Tuple[int, int] = (20, 5)
    ) -> plt.Figure:
        """
        Visualize individual attention heads.
        
        Args:
            attention_map: Attention tensor with shape (heads, tokens, tokens)
            num_heads: Number of heads to visualize
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if attention_map.ndim == 4:
            attention_map = attention_map[0]  # Take first batch
        
        num_heads = min(num_heads, attention_map.shape[0])
        fig, axes = plt.subplots(1, num_heads, figsize=figsize)
        
        if num_heads == 1:
            axes = [axes]
        
        for i in range(num_heads):
            attn_np = attention_map[i].cpu().numpy()
            sns.heatmap(
                attn_np,
                cmap='viridis',
                ax=axes[i],
                cbar=True,
                square=True,
                xticklabels=False,
                yticklabels=False
            )
            axes[i].set_title(f'Head {i+1}')
        
        plt.tight_layout()
        return fig
    
    def visualize_attention_trajectory(
        self,
        x: torch.Tensor,
        timesteps: List[int] = [0, 250, 500, 750, 999],
        layer_name: Optional[str] = None,
        save_dir: Optional[str] = None
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Visualize how attention patterns change across diffusion timesteps.
        
        Args:
            x: Input tensor
            timesteps: List of timesteps to analyze
            layer_name: Specific layer to track (or None for all)
            save_dir: Optional directory to save visualizations
            
        Returns:
            Dictionary mapping timesteps to attention maps
        """
        device = x.device
        results = {}
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for t_val in timesteps:
            t = torch.full((x.shape[0],), t_val, device=device, dtype=torch.long)
            
            # Add noise
            if hasattr(self.model, 'q_sample'):
                x_t = self.model.q_sample(x, t)
            else:
                noise = torch.randn_like(x)
                alpha = 1 - t_val / 1000
                x_t = alpha * x + (1 - alpha) * noise
            
            # Extract attention maps
            attention_maps = self.extract_attention_maps(x_t, t)
            
            # Filter by layer name if specified
            if layer_name:
                attention_maps = {
                    k: v for k, v in attention_maps.items()
                    if layer_name in k
                }
            
            results[t_val] = attention_maps
            
            # Visualize if save_dir provided
            if save_dir:
                for name, attn_map in attention_maps.items():
                    safe_name = name.replace(".", "_").replace("/", "_")
                    save_path = Path(save_dir) / f"attention_t{t_val}_{safe_name}.png"
                    
                    fig = self.visualize_attention_map(
                        attn_map,
                        title=f"Attention at t={t_val}: {name}",
                        save_path=str(save_path)
                    )
                    plt.close(fig)
        
        return results
    
    def compute_attention_rollout(
        self,
        attention_maps: Dict[str, torch.Tensor],
        discard_ratio: float = 0.1
    ) -> torch.Tensor:
        """
        Compute attention rollout to track information flow.
        
        Args:
            attention_maps: Dictionary of attention maps from different layers
            discard_ratio: Ratio of low attention values to discard
            
        Returns:
            Rolled out attention map
        """
        # Start with identity matrix
        maps = list(attention_maps.values())
        if len(maps) == 0:
            return None
        
        # Average attention heads
        result = None
        for attn_map in maps:
            if attn_map.ndim == 4:
                attn_map = attn_map[0].mean(0)  # (tokens, tokens)
            elif attn_map.ndim == 3:
                attn_map = attn_map.mean(0)
            
            # Discard lowest attention values
            flat = attn_map.flatten()
            threshold = torch.quantile(flat, discard_ratio)
            attn_map = torch.where(attn_map < threshold, torch.zeros_like(attn_map), attn_map)
            
            # Normalize
            attn_map = attn_map / (attn_map.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Accumulate
            if result is None:
                result = attn_map
            else:
                result = torch.matmul(attn_map, result)
        
        return result
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


if __name__ == "__main__":
    # Test Attention Visualizer
    from src.models.diffusion import DDPM
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = DDPM(image_size=64, channels=3, timesteps=1000).to(device)
    
    # Create visualizer
    visualizer = AttentionVisualizer(model)
    
    # Test on random input
    x = torch.randn(2, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    
    # Extract attention maps
    attention_maps = visualizer.extract_attention_maps(x, t)
    print(f"Extracted attention maps from layers: {list(attention_maps.keys())}")
    
    # Cleanup
    visualizer.remove_hooks()

