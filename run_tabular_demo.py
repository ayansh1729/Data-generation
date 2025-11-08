"""
Complete Tabular Data Generation Demo
Run this to see the full workflow and results
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# Ensure output directory exists
os.makedirs('outputs', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("TABULAR DATA GENERATION WITH DIFFUSION MODELS")
print("="*70)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load data
print("\nLoading Breast Cancer Wisconsin Dataset...")
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print(f"   Samples: {X.shape[0]}")
print(f"   Features: {X.shape[1]}")
print(f"   Malignant: {(y==0).sum()}, Benign: {(y==1).sum()}")

# Normalize
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Create dataset
class TabularDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = TabularDataset(X_normalized)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

# Create model
print("\nCreating Tabular Diffusion Model...")
from src.models.tabular_diffusion import TabularDiffusionModel

input_dim = X.shape[1]
model = TabularDiffusionModel(
    input_dim=input_dim,
    hidden_dims=(256, 512, 512, 256),
    time_emb_dim=128,
    dropout=0.1
).to(device)

# DDPM wrapper
class TabularDDPM(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # Cosine schedule
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
    
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        x_noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        
        predicted = self.model(x_noisy, t)
        loss = nn.functional.mse_loss(predicted, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(self, num_samples, dim):
        device = next(self.parameters()).device
        x = torch.randn(num_samples, dim, device=device)
        
        for t in tqdm(reversed(range(self.timesteps)), desc='Sampling'):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = self.model(x, t_batch)
            
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            # Compute mean of x_{t-1}
            model_mean = (x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            if t > 0:
                noise = torch.randn_like(x)
                # Posterior variance
                alpha_cumprod_t_prev = self.alphas_cumprod[t-1]
                posterior_variance = beta_t * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)
                x = model_mean + torch.sqrt(posterior_variance) * noise
                # Clip to prevent explosion (important for tabular data)
                x = torch.clamp(x, -10, 10)
            else:
                x = model_mean
        
        return x

ddpm = TabularDDPM(model, timesteps=1000).to(device)
num_params = sum(p.numel() for p in ddpm.parameters())
print(f"   Parameters: {num_params:,}")

# Training
print("\nTraining (100 epochs)...")
optimizer = torch.optim.AdamW(ddpm.parameters(), lr=1e-3)
num_epochs = 100
losses = []

ddpm.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in dataloader:
        batch = batch.to(device)
        loss = ddpm(batch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    avg_loss = np.mean(epoch_losses)
    losses.append(avg_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"   Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

print("\nTraining complete!")

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(losses, linewidth=2, color='steelblue')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss', fontweight='bold', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/training_loss.png', dpi=150, bbox_inches='tight')
print("   Saved: outputs/training_loss.png")

# Generate synthetic data
print("\nGenerating synthetic data...")
ddpm.eval()
synthetic_data = ddpm.sample(X.shape[0], input_dim)
synthetic_data_np = synthetic_data.cpu().numpy()
synthetic_data_denorm = scaler.inverse_transform(synthetic_data_np)

print(f"   Generated {synthetic_data_denorm.shape[0]} samples")

# PCA comparison
print("\nCreating visualizations...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
synthetic_pca = pca.transform(synthetic_data_denorm)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, edgecolors='k', s=50)
axes[0].set_title('Real Data', fontweight='bold', fontsize=14)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=11)
axes[0].grid(alpha=0.3)

axes[1].scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], alpha=0.6, edgecolors='k', s=50, color='coral')
axes[1].set_title('Synthetic Data', fontweight='bold', fontsize=14)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=11)
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/pca_comparison.png', dpi=150, bbox_inches='tight')
print("   Saved: outputs/pca_comparison.png")

# Distribution comparison
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i in range(9):
    axes[i].hist(X[:, i], bins=30, alpha=0.5, label='Real', color='blue', edgecolor='black')
    axes[i].hist(synthetic_data_denorm[:, i], bins=30, alpha=0.5, label='Synthetic', color='red', edgecolor='black')
    axes[i].set_title(f"{feature_names[i]}", fontsize=10, fontweight='bold')
    axes[i].legend(fontsize=9)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/distributions.png', dpi=150, bbox_inches='tight')
print("   Saved: outputs/distributions.png")

# Save CSV
synthetic_df = pd.DataFrame(synthetic_data_denorm, columns=feature_names)
synthetic_df.to_csv('outputs/synthetic_data.csv', index=False)
print("   Saved: outputs/synthetic_data.csv")

# Statistics
print("\nStatistical Comparison:")
print("="*70)
print(f"{'Feature':<35} {'Real Mean':>12} {'Synth Mean':>12} {'Diff %':>10}")
print("="*70)

total_diff = 0
for i in range(min(10, len(feature_names))):
    real_mean = X[:, i].mean()
    synth_mean = synthetic_data_denorm[:, i].mean()
    diff = abs((real_mean - synth_mean) / real_mean * 100)
    total_diff += diff
    print(f"{feature_names[i][:34]:<35} {real_mean:12.2f} {synth_mean:12.2f} {diff:9.2f}%")

print("="*70)
print(f"Average difference: {total_diff/10:.2f}%")
print("="*70)

print("\nCOMPLETE! Check the outputs/ folder for:")
print("   - training_loss.png")
print("   - pca_comparison.png")
print("   - distributions.png")
print("   - synthetic_data.csv")
print("="*70)

