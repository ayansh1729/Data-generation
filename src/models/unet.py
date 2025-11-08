"""
U-Net Architecture for Diffusion Models

Implements a U-Net with residual blocks, attention mechanisms, and time embeddings
for denoising diffusion probabilistic models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from einops import rearrange


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding and group normalization."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        groups: int = 8
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block with group normalization."""
    
    def __init__(self, channels: int, num_heads: int = 4, groups: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)',
            heads=self.num_heads, qkv=3
        )
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhci,bhcj->bhij', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhij,bhcj->bhci', attn, v)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', h=h, w=w)
        out = self.proj(out)
        
        return out + residual


class DownBlock(nn.Module):
    """Downsampling block with residual blocks and optional attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_blocks: int = 2,
        use_attention: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout
            )
            for i in range(num_blocks)
        ])
        
        self.attention = nn.ModuleList([
            AttentionBlock(out_channels) if use_attention else nn.Identity()
            for _ in range(num_blocks)
        ])
        
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_connections = []
        
        for block, attn in zip(self.blocks, self.attention):
            x = block(x, time_emb)
            x = attn(x)
            skip_connections.append(x)
        
        x = self.downsample(x)
        
        return x, skip_connections


class UpBlock(nn.Module):
    """Upsampling block with residual blocks and optional attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_blocks: int = 2,
        use_attention: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            ResidualBlock(
                in_channels + out_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout
            )
            for i in range(num_blocks)
        ])
        
        self.attention = nn.ModuleList([
            AttentionBlock(out_channels) if use_attention else nn.Identity()
            for _ in range(num_blocks)
        ])
        
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor],
        time_emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)
        
        for i, (block, attn) in enumerate(zip(self.blocks, self.attention)):
            if i == 0 and len(skip_connections) > 0:
                x = torch.cat([x, skip_connections.pop()], dim=1)
            x = block(x, time_emb)
            x = attn(x)
        
        return x


class UNet(nn.Module):
    """
    U-Net backbone for diffusion models.
    
    Args:
        image_size: Input image size
        in_channels: Number of input channels
        out_channels: Number of output channels
        dim: Base channel dimension
        dim_mults: Channel multipliers for each resolution level
        num_res_blocks: Number of residual blocks per level
        attention_resolutions: Resolutions at which to apply attention
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 64,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        
        # Time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding=3)
        
        # Compute dimensions for each level
        dims = [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # Downsampling
        self.downs = nn.ModuleList([])
        current_res = image_size
        
        for i, (dim_in, dim_out) in enumerate(in_out):
            current_res //= 2
            use_attention = current_res in attention_resolutions
            
            self.downs.append(
                DownBlock(
                    dim_in,
                    dim_out,
                    time_dim,
                    num_res_blocks,
                    use_attention,
                    dropout
                )
            )
        
        # Middle
        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_dim, dropout)
        self.mid_attn = AttentionBlock(mid_dim)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_dim, dropout)
        
        # Upsampling
        self.ups = nn.ModuleList([])
        
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            current_res *= 2
            use_attention = current_res in attention_resolutions
            
            self.ups.append(
                UpBlock(
                    dim_out,
                    dim_in,
                    time_dim,
                    num_res_blocks,
                    use_attention,
                    dropout
                )
            )
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            time: Time step tensor of shape (B,)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Time embedding
        t = self.time_mlp(time)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Store skip connections
        all_skip_connections = []
        
        # Downsampling
        for down in self.downs:
            x, skip_connections = down(x, t)
            all_skip_connections.extend(skip_connections)
        
        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # Upsampling
        for up in self.ups:
            # Get skip connections for this level
            level_skips = [
                all_skip_connections.pop()
                for _ in range(len(up.blocks))
                if all_skip_connections
            ]
            level_skips.reverse()
            x = up(x, level_skips, t)
        
        # Final convolution
        return self.final_conv(x)


if __name__ == "__main__":
    # Test the U-Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet(
        image_size=64,
        in_channels=3,
        out_channels=3,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.1
    ).to(device)
    
    # Test forward pass
    x = torch.randn(4, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (4,)).to(device)
    
    out = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

