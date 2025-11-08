"""
SHAP (SHapley Additive exPlanations) Explainer for Diffusion Models

Provides model-agnostic feature importance explanations for diffusion models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, List, Tuple
import matplotlib.pyplot as plt
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


class SHAPExplainer:
    """
    SHAP-based explainer for diffusion models.
    
    Args:
        model: Diffusion model to explain
        background_data: Background dataset for SHAP baseline
        num_samples: Number of samples for SHAP estimation
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None,
        num_samples: int = 100
    ):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        self.model = model
        self.num_samples = num_samples
        
        if background_data is None:
            # Create random background
            device = next(model.parameters()).device
            self.background_data = torch.randn(
                num_samples,
                model.channels,
                model.image_size,
                model.image_size
            ).to(device)
        else:
            self.background_data = background_data
    
    def create_prediction_function(self, timestep: int) -> Callable:
        """
        Create a prediction function for a specific timestep.
        
        Args:
            timestep: The timestep to analyze
            
        Returns:
            Prediction function compatible with SHAP
        """
        device = next(self.model.parameters()).device
        
        def predict(x: np.ndarray) -> np.ndarray:
            """Prediction function for SHAP."""
            self.model.eval()
            
            # Convert to tensor
            if isinstance(x, np.ndarray):
                x_tensor = torch.from_numpy(x).float().to(device)
            else:
                x_tensor = x.to(device)
            
            batch_size = x_tensor.shape[0]
            t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            
            with torch.no_grad():
                if hasattr(self.model, 'model'):
                    output = self.model.model(x_tensor, t)
                else:
                    output = self.model(x_tensor, t)
            
            # Return mean prediction per sample
            return output.cpu().numpy().reshape(batch_size, -1).mean(axis=1)
        
        return predict
    
    def explain_sample(
        self,
        x: torch.Tensor,
        timestep: int,
        method: str = "partition"
    ) -> "shap.Explanation":
        """
        Explain a sample's prediction at a specific timestep.
        
        Args:
            x: Input tensor
            timestep: Timestep to analyze
            method: SHAP method ('partition', 'kernel', 'gradient')
            
        Returns:
            SHAP explanation object
        """
        predict_fn = self.create_prediction_function(timestep)
        
        # Prepare data
        x_np = x.cpu().numpy()
        background_np = self.background_data.cpu().numpy()
        
        # Create explainer based on method
        if method == "gradient":
            # Gradient-based explainer (faster, requires gradients)
            explainer = shap.GradientExplainer(
                lambda x: torch.tensor(predict_fn(x.cpu().numpy())).to(x.device),
                self.background_data
            )
        elif method == "kernel":
            # Kernel SHAP (model-agnostic, slower)
            explainer = shap.KernelExplainer(predict_fn, background_np[:50])
        else:
            # Partition explainer (fast, hierarchical)
            explainer = shap.PartitionExplainer(predict_fn, background_np[:50])
        
        # Compute SHAP values
        shap_values = explainer(x_np)
        
        return shap_values
    
    def visualize_shap_values(
        self,
        shap_values: "shap.Explanation",
        sample_idx: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4)
    ) -> plt.Figure:
        """
        Visualize SHAP values for a sample.
        
        Args:
            shap_values: SHAP explanation object
            sample_idx: Which sample to visualize
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        img = shap_values.data[sample_idx]
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # SHAP values
        shap_img = shap_values.values[sample_idx]
        if shap_img.ndim == 3:
            shap_img = np.transpose(shap_img, (1, 2, 0))
            shap_img = shap_img.mean(axis=-1)  # Average across channels
        
        im = axes[1].imshow(shap_img, cmap='RdBu', vmin=-np.abs(shap_img).max(), vmax=np.abs(shap_img).max())
        axes[1].set_title('SHAP Values')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        overlay = img.copy()
        if overlay.shape[-1] == 3:
            overlay = overlay.mean(axis=-1)
        
        axes[2].imshow(overlay, cmap='gray', alpha=0.5)
        axes[2].imshow(np.abs(shap_img), cmap='Reds', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def explain_trajectory(
        self,
        x: torch.Tensor,
        timesteps: List[int] = [0, 250, 500, 750, 999],
        save_dir: Optional[str] = None
    ) -> Dict[int, "shap.Explanation"]:
        """
        Explain predictions across multiple timesteps.
        
        Args:
            x: Input tensor
            timesteps: List of timesteps to analyze
            save_dir: Optional directory to save visualizations
            
        Returns:
            Dictionary mapping timesteps to SHAP explanations
        """
        explanations = {}
        
        if save_dir:
            from pathlib import Path
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for t in timesteps:
            print(f"Computing SHAP values for timestep {t}...")
            shap_values = self.explain_sample(x, t)
            explanations[t] = shap_values
            
            if save_dir:
                save_path = Path(save_dir) / f"shap_t{t}.png"
                self.visualize_shap_values(shap_values, save_path=str(save_path))
                plt.close()
        
        return explanations


# Simplified SHAP implementation for when shap library is not available
class SimpleSHAPExplainer:
    """
    Simplified SHAP-like feature importance using gradients.
    Works when the full SHAP library is not available.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def compute_gradient_importance(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature importance using input gradients.
        
        Args:
            x: Input tensor
            t: Timestep tensor
            
        Returns:
            Importance map
        """
        x.requires_grad = True
        
        # Forward pass
        if hasattr(self.model, 'model'):
            output = self.model.model(x, t)
        else:
            output = self.model(x, t)
        
        # Compute gradients
        self.model.zero_grad()
        target = output.mean()
        target.backward()
        
        # Importance = |gradient * input|
        importance = (x.grad * x).abs()
        
        return importance.detach()
    
    def visualize_importance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize feature importance."""
        importance = self.compute_gradient_importance(x, t)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original
        img = x[sample_idx].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Importance
        imp = importance[sample_idx].cpu().numpy()
        if imp.shape[0] > 1:
            imp = imp.mean(axis=0)
        else:
            imp = imp[0]
        
        im = axes[1].imshow(imp, cmap='hot')
        axes[1].set_title('Feature Importance')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        if img.shape[-1] == 3:
            img_gray = img.mean(axis=-1)
        else:
            img_gray = img.squeeze()
        
        axes[2].imshow(img_gray, cmap='gray', alpha=0.5)
        axes[2].imshow(imp, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Test SHAP Explainer
    from src.models.diffusion import DDPM
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = DDPM(image_size=32, channels=3, timesteps=1000).to(device)
    
    # Test simplified explainer (always available)
    simple_explainer = SimpleSHAPExplainer(model)
    
    x = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    
    fig = simple_explainer.visualize_importance(x, t, sample_idx=0)
    print("Simple explainer visualization complete")
    plt.close(fig)
    
    # Test full SHAP if available
    if SHAP_AVAILABLE:
        explainer = SHAPExplainer(model, num_samples=50)
        print("Full SHAP explainer initialized")

