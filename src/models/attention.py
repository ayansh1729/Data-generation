"""
Attention mechanisms for the U-Net architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Args:
        dim: Input/output dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, N, D)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (B, N, D)
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        
        return out


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for conditioning.
    
    Args:
        query_dim: Query dimension
        context_dim: Context dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Query tensor of shape (B, N, D)
            context: Context tensor of shape (B, M, C)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (B, N, D)
        """
        B, N, D = x.shape
        M = context.shape[1]
        
        # Compute Q from x, K and V from context
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.to_k(context).reshape(B, M, self.num_heads, self.head_dim)
        v = self.to_v(context).reshape(B, M, self.num_heads, self.head_dim)
        
        q = q.permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, num_heads, M, head_dim)
        v = v.permute(0, 2, 1, 3)  # (B, num_heads, M, head_dim)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, M)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        
        return out


if __name__ == "__main__":
    # Test attention mechanisms
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test MultiHeadAttention
    mha = MultiHeadAttention(dim=512, num_heads=8).to(device)
    x = torch.randn(2, 10, 512).to(device)
    out = mha(x)
    print(f"MultiHeadAttention - Input: {x.shape}, Output: {out.shape}")
    
    # Test CrossAttention
    ca = CrossAttention(query_dim=512, context_dim=768, num_heads=8).to(device)
    x = torch.randn(2, 10, 512).to(device)
    context = torch.randn(2, 20, 768).to(device)
    out = ca(x, context)
    print(f"CrossAttention - Query: {x.shape}, Context: {context.shape}, Output: {out.shape}")

