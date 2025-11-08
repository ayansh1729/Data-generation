"""
Tabular Diffusion Model - specialized architecture for tabular data generation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TabularDiffusionModel(nn.Module):
    """
    Diffusion model specialized for tabular data.
    Uses a simple MLP architecture instead of U-Net.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 512, 512, 256),
        time_emb_dim: int = 128,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize Tabular Diffusion Model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: Tuple of hidden layer dimensions
            time_emb_dim: Dimension of time embedding
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(
                ResidualBlock(
                    prev_dim,
                    hidden_dim,
                    time_emb_dim,
                    dropout,
                    use_batch_norm
                )
            )
            prev_dim = hidden_dim
        
        self.blocks = nn.ModuleList(layers)
        
        # Output layer
        self.output = nn.Linear(prev_dim, input_dim)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, D)
            t: Time step tensor of shape (B,)
            
        Returns:
            Predicted noise of shape (B, D)
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Process through blocks
        h = x
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Output
        return self.output(h)


class ResidualBlock(nn.Module):
    """Residual block for tabular data."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.time_proj = nn.Linear(time_emb_dim, out_dim)
        
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = self.block1(x)
        
        # Add time embedding
        h = h + self.time_proj(t_emb)
        
        h = self.block2(h)
        
        # Residual connection
        return h + self.residual(x)


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time step embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings.
        
        Args:
            time: Time steps of shape (B,)
            
        Returns:
            Embeddings of shape (B, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


if __name__ == "__main__":
    # Test the model
    batch_size = 32
    input_dim = 20
    
    model = TabularDiffusionModel(
        input_dim=input_dim,
        hidden_dims=(256, 512, 512, 256),
        time_emb_dim=128,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, input_dim)
    t = torch.randint(0, 1000, (batch_size,))
    
    output = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

