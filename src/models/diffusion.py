"""
Denoising Diffusion Probabilistic Model (DDPM) Implementation

Based on the paper: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from tqdm import tqdm

from src.models.unet import UNet


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear schedule from Ho et al., 2020
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def sigmoid_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Sigmoid schedule
    """
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.
    
    Args:
        model: Noise prediction network (U-Net)
        image_size: Size of input images
        timesteps: Number of diffusion timesteps
        noise_schedule: Type of noise schedule ('linear', 'cosine', 'sigmoid')
        objective: Training objective ('pred_noise', 'pred_x0', 'pred_v')
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        image_size: int = 64,
        channels: int = 3,
        timesteps: int = 1000,
        noise_schedule: str = "cosine",
        objective: str = "pred_noise",
        **kwargs
    ):
        super().__init__()
        
        if model is None:
            self.model = UNet(
                image_size=image_size,
                in_channels=channels,
                out_channels=channels,
                **kwargs
            )
        else:
            self.model = model
            
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.objective = objective
        
        # Define beta schedule
        if noise_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif noise_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif noise_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")
        
        # Pre-compute diffusion parameters
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register as buffers (non-trainable parameters)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_start: Clean images
            t: Timesteps
            noise: Optional pre-generated noise
            
        Returns:
            Noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        """
        return (
            self.sqrt_recip_alphas_cumprod[t][:, None, None, None] * x_t -
            self.sqrt_recipm1_alphas_cumprod[t][:, None, None, None] * noise
        )
    
    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the posterior mean and variance q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            self.posterior_mean_coef1[t][:, None, None, None] * x_start +
            self.posterior_mean_coef2[t][:, None, None, None] * x_t
        )
        posterior_variance = self.posterior_variance[t][:, None, None, None]
        posterior_log_variance = self.posterior_log_variance_clipped[t][:, None, None, None]
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of p(x_{t-1} | x_t)
        """
        # Predict noise
        model_output = self.model(x, t)
        
        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
        elif self.objective == "pred_x0":
            x_start = model_output
        else:
            raise NotImplementedError(f"Objective {self.objective} not implemented")
        
        # Clip predicted x_start
        x_start = torch.clamp(x_start, -1.0, 1.0)
        
        # Compute posterior
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start, x, t)
        
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t)
        """
        model_mean, _, model_log_variance = self.p_mean_variance(x, t)
        noise = torch.randn_like(x)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def sample(self, batch_size: int = 16, return_all_timesteps: bool = False) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Args:
            batch_size: Number of samples to generate
            return_all_timesteps: If True, return samples at all timesteps
            
        Returns:
            Generated samples
        """
        device = next(self.parameters()).device
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        if return_all_timesteps:
            imgs = [img]
        
        # Iteratively denoise
        for t in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps):
            img = self.p_sample(img, torch.full((batch_size,), t, device=device, dtype=torch.long))
            
            if return_all_timesteps:
                imgs.append(img)
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=1)
        
        return img
    
    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            x: Clean images
            noise: Optional pre-generated noise
            
        Returns:
            Dictionary containing loss and other metrics
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # Generate noise
        if noise is None:
            noise = torch.randn_like(x)
        
        # Forward diffusion
        x_noisy = self.q_sample(x, t, noise)
        
        # Predict noise
        predicted = self.model(x_noisy, t)
        
        # Compute loss
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x
        else:
            raise NotImplementedError(f"Objective {self.objective} not implemented")
        
        loss = F.mse_loss(predicted, target)
        
        return {
            "loss": loss,
            "predicted": predicted,
            "target": target,
            "x_noisy": x_noisy,
            "t": t
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "image_size": self.image_size,
                "channels": self.channels,
                "timesteps": self.timesteps,
                "objective": self.objective,
            }
        }, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cuda"):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device)


if __name__ == "__main__":
    # Test DDPM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DDPM(
        image_size=64,
        channels=3,
        timesteps=1000,
        noise_schedule="cosine"
    ).to(device)
    
    # Test training forward pass
    x = torch.randn(4, 3, 64, 64).to(device)
    outputs = model(x)
    
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test sampling
    samples = model.sample(batch_size=4)
    print(f"Generated samples shape: {samples.shape}")

