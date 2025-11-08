"""
Tabular Data Generation Example

This script demonstrates how to generate synthetic tabular data using
diffusion models. Can be run directly or converted to a notebook.
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

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def main():
    print("="*70)
    print("🎲 TABULAR DATA GENERATION WITH DIFFUSION MODELS")
    print("="*70)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    # Load data
    print("\n📊 Loading Breast Cancer Wisconsin Dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    print(f"   Samples: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # Normalize
    print("\n🔧 Normalizing data...")
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Create dataset
    class SimpleTabularDataset(Dataset):
        def __init__(self, data):
            self.data = torch.FloatTensor(data)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = SimpleTabularDataset(X_normalized)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    
    # Create model (import from project)
    print("\n🏗️ Creating Tabular Diffusion Model...")
    from src.models.tabular_diffusion import TabularDiffusionModel
    
    input_dim = X.shape[1]
    model = TabularDiffusionModel(
        input_dim=input_dim,
        hidden_dims=(256, 512, 512, 256),
        time_emb_dim=128,
        dropout=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    # Create diffusion wrapper
    print("\n🔧 Creating DDPM wrapper...")
    
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
            
            # Random timesteps
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
            noise = torch.randn_like(x)
            
            # Add noise
            sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            x_noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise
            
            # Predict
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
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = (x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)
                x = x + torch.sqrt(beta_t) * noise
            
            return x
    
    ddpm = TabularDDPM(model, timesteps=1000).to(device)
    
    # Training
    print("\n🎯 Training...")
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=1e-3)
    num_epochs = 50
    
    ddpm.train()
    for epoch in range(num_epochs):
        losses = []
        for batch in dataloader:
            batch = batch.to(device)
            loss = ddpm(batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{num_epochs} - Loss: {np.mean(losses):.4f}")
    
    print("\n✅ Training complete!")
    
    # Generate synthetic data
    print("\n🎲 Generating synthetic data...")
    ddpm.eval()
    synthetic_data = ddpm.sample(X.shape[0], input_dim)
    synthetic_data_np = synthetic_data.cpu().numpy()
    synthetic_data_denorm = scaler.inverse_transform(synthetic_data_np)
    
    print(f"   Generated {synthetic_data_denorm.shape[0]} samples")
    
    # Visualize with PCA
    print("\n📊 Visualizing with PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    synthetic_pca = pca.transform(synthetic_data_denorm)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, edgecolors='k')
    axes[0].set_title('Real Data', fontweight='bold', fontsize=14)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    axes[0].grid(alpha=0.3)
    
    axes[1].scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], alpha=0.6, edgecolors='k', color='red')
    axes[1].set_title('Synthetic Data', fontweight='bold', fontsize=14)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tabular_pca_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: tabular_pca_comparison.png")
    
    # Compare distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(6):
        axes[i].hist(X[:, i], bins=30, alpha=0.5, label='Real', color='blue', edgecolor='black')
        axes[i].hist(synthetic_data_denorm[:, i], bins=30, alpha=0.5, label='Synthetic', color='red', edgecolor='black')
        axes[i].set_title(f"{feature_names[i]}", fontsize=10)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('tabular_distributions.png', dpi=150, bbox_inches='tight')
    print("   Saved: tabular_distributions.png")
    
    # Save synthetic data
    synthetic_df = pd.DataFrame(synthetic_data_denorm, columns=feature_names)
    synthetic_df.to_csv('synthetic_tabular_data.csv', index=False)
    print("   Saved: synthetic_tabular_data.csv")
    
    # Statistics comparison
    print("\n📈 Statistical Comparison (first 5 features):")
    for i in range(5):
        real_mean = X[:, i].mean()
        synth_mean = synthetic_data_denorm[:, i].mean()
        diff = abs((real_mean - synth_mean) / real_mean * 100)
        print(f"   {feature_names[i][:30]:30s} - Mean diff: {diff:6.2f}%")
    
    print("\n" + "="*70)
    print("✅ COMPLETE! Check the generated files:")
    print("   - tabular_pca_comparison.png")
    print("   - tabular_distributions.png")
    print("   - synthetic_tabular_data.csv")
    print("="*70)


if __name__ == "__main__":
    main()

