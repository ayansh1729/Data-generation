"""
Complete Tabular Data Generation with Comprehensive Evaluation
Includes Wasserstein distance, classifier accuracy, and detailed metrics
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from scipy.stats import wasserstein_distance, ks_2samp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# Ensure output directory exists
os.makedirs('outputs', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("TABULAR DATA GENERATION WITH COMPREHENSIVE EVALUATION")
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

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Normalize
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Create dataset
class TabularDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = TabularDataset(X_train_normalized)
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
        
        for t in tqdm(reversed(range(self.timesteps)), desc='Sampling', leave=False):
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
print("\nTraining (150 epochs for better quality)...")
optimizer = torch.optim.AdamW(ddpm.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
num_epochs = 150
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
    
    scheduler.step()
    avg_loss = np.mean(epoch_losses)
    losses.append(avg_loss)
    
    if (epoch + 1) % 30 == 0:
        print(f"   Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

print("\nTraining complete!")
print(f"   Final loss: {losses[-1]:.4f}")

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(losses, linewidth=2, color='steelblue')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss', fontweight='bold', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/training_loss.png', dpi=150, bbox_inches='tight')
print("\n   Saved: outputs/training_loss.png")

# Generate synthetic data
print("\nGenerating synthetic data...")
ddpm.eval()
synthetic_data = ddpm.sample(X_train.shape[0], input_dim)
synthetic_data_np = synthetic_data.cpu().numpy()
synthetic_data_denorm = scaler.inverse_transform(synthetic_data_np)

print(f"   Generated {synthetic_data_denorm.shape[0]} samples")

# ============================================================================
# EVALUATION METRICS
# ============================================================================
print("\n" + "="*70)
print("EVALUATION METRICS")
print("="*70)

# 1. Wasserstein Distance (Earth Mover's Distance)
print("\n1. WASSERSTEIN DISTANCE (lower is better):")
print("-"*70)
wasserstein_distances = []
for i in range(X.shape[1]):
    wd = wasserstein_distance(X_train[:, i], synthetic_data_denorm[:, i])
    wasserstein_distances.append(wd)
    if i < 10:  # Show first 10
        print(f"   {feature_names[i][:40]:40s}: {wd:10.4f}")

avg_wasserstein = np.mean(wasserstein_distances)
print(f"\n   Average Wasserstein Distance: {avg_wasserstein:.4f}")
print(f"   {'   (0 = perfect, lower is better)'}")

# 2. Kolmogorov-Smirnov Test
print("\n2. KOLMOGOROV-SMIRNOV TEST (p-value, higher is better):")
print("-"*70)
ks_pvalues = []
for i in range(min(10, X.shape[1])):
    statistic, pvalue = ks_2samp(X_train[:, i], synthetic_data_denorm[:, i])
    ks_pvalues.append(pvalue)
    status = "PASS" if pvalue > 0.05 else "FAIL"
    print(f"   {feature_names[i][:40]:40s}: p={pvalue:.4f} [{status}]")

print(f"\n   Passed: {sum(p > 0.05 for p in ks_pvalues)}/{len(ks_pvalues)} features")

# 3. Statistical Comparison
print("\n3. STATISTICAL COMPARISON:")
print("-"*70)
print(f"{'Feature':<35} {'Real Mean':>12} {'Synth Mean':>12} {'Diff %':>10}")
print("-"*70)

total_diff = 0
for i in range(min(10, len(feature_names))):
    real_mean = X_train[:, i].mean()
    synth_mean = synthetic_data_denorm[:, i].mean()
    diff = abs((real_mean - synth_mean) / (real_mean + 1e-8) * 100)
    total_diff += diff
    print(f"{feature_names[i][:34]:<35} {real_mean:12.2f} {synth_mean:12.2f} {diff:9.2f}%")

print("-"*70)
print(f"Average Mean Difference: {total_diff/10:.2f}%")

# 4. Classifier Performance (TRAIN ON REAL, TEST ON SYNTHETIC vs TEST ON REAL)
print("\n4. CLASSIFIER PERFORMANCE (TRTR vs TSTR):")
print("-"*70)
print("Training classifiers on REAL data...")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Test on Real test set (TRTR - Train Real, Test Real)
rf_acc_real = accuracy_score(y_test, rf_model.predict(X_test))
lr_acc_real = accuracy_score(y_test, lr_model.predict(X_test))

print(f"\n   Random Forest (TRTR): {rf_acc_real*100:.2f}%")
print(f"   Logistic Regression (TRTR): {lr_acc_real*100:.2f}%")

# Generate labels for synthetic data (predict using trained model)
y_synthetic = rf_model.predict(synthetic_data_denorm)

# Test on Synthetic data (TSTR - Train Real, Test Synthetic)
# We'll evaluate how well the synthetic data distribution matches by predicting labels
print(f"\n   Synthetic data label distribution:")
print(f"      Predicted Malignant: {(y_synthetic==0).sum()} ({(y_synthetic==0).sum()/len(y_synthetic)*100:.1f}%)")
print(f"      Predicted Benign: {(y_synthetic==1).sum()} ({(y_synthetic==1).sum()/len(y_synthetic)*100:.1f}%)")
print(f"   Real train label distribution:")
print(f"      Malignant: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
print(f"      Benign: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")

# 5. Train classifier on SYNTHETIC, test on REAL (TSTR)
print("\n5. GENERALIZATION TEST (Train on Synthetic, Test on Real):")
print("-"*70)

rf_synth = RandomForestClassifier(n_estimators=100, random_state=42)
rf_synth.fit(synthetic_data_denorm, y_synthetic)

lr_synth = LogisticRegression(max_iter=1000, random_state=42)
lr_synth.fit(synthetic_data_denorm, y_synthetic)

rf_acc_tstr = accuracy_score(y_test, rf_synth.predict(X_test))
lr_acc_tstr = accuracy_score(y_test, lr_synth.predict(X_test))

print(f"   Random Forest (TSTR): {rf_acc_tstr*100:.2f}%")
print(f"   Logistic Regression (TSTR): {lr_acc_tstr*100:.2f}%")

print(f"\n   Performance Gap:")
print(f"      Random Forest: {abs(rf_acc_real - rf_acc_tstr)*100:.2f}% drop")
print(f"      Logistic Regression: {abs(lr_acc_real - lr_acc_tstr)*100:.2f}% drop")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# PCA comparison
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
synthetic_pca = pca.transform(synthetic_data_denorm)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.6, edgecolors='k', s=50)
axes[0].set_title('Real Data', fontweight='bold', fontsize=14)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=11)
axes[0].grid(alpha=0.3)

axes[1].scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], c=y_synthetic, cmap='viridis', alpha=0.6, edgecolors='k', s=50)
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
    axes[i].hist(X_train[:, i], bins=30, alpha=0.5, label='Real', color='blue', edgecolor='black')
    axes[i].hist(synthetic_data_denorm[:, i], bins=30, alpha=0.5, label='Synthetic', color='red', edgecolor='black')
    axes[i].set_title(f"{feature_names[i]}", fontsize=10, fontweight='bold')
    axes[i].legend(fontsize=9)
    axes[i].grid(alpha=0.3)
    
    # Add Wasserstein distance
    wd = wasserstein_distances[i]
    axes[i].text(0.95, 0.95, f'WD: {wd:.2f}', 
                transform=axes[i].transAxes, 
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('outputs/distributions.png', dpi=150, bbox_inches='tight')
print("   Saved: outputs/distributions.png")

# Wasserstein distances plot
plt.figure(figsize=(12, 6))
plt.bar(range(len(wasserstein_distances)), wasserstein_distances, color='steelblue', edgecolor='black')
plt.xlabel('Feature Index', fontsize=12)
plt.ylabel('Wasserstein Distance', fontsize=12)
plt.title('Wasserstein Distance per Feature', fontweight='bold', fontsize=14)
plt.axhline(y=avg_wasserstein, color='r', linestyle='--', label=f'Average: {avg_wasserstein:.2f}')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('outputs/wasserstein_distances.png', dpi=150, bbox_inches='tight')
print("   Saved: outputs/wasserstein_distances.png")

# Accuracy comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = ['Random Forest', 'Logistic Regression']
trtr_scores = [rf_acc_real*100, lr_acc_real*100]
tstr_scores = [rf_acc_tstr*100, lr_acc_tstr*100]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, trtr_scores, width, label='TRTR (Real→Real)', color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2, tstr_scores, width, label='TSTR (Synth→Real)', color='coral', edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Classifier Performance Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/accuracy_comparison.png', dpi=150, bbox_inches='tight')
print("   Saved: outputs/accuracy_comparison.png")

# Save synthetic data
synthetic_df = pd.DataFrame(synthetic_data_denorm, columns=feature_names)
synthetic_df['predicted_label'] = y_synthetic
synthetic_df.to_csv('outputs/synthetic_data.csv', index=False)
print("   Saved: outputs/synthetic_data.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"\nQUALITY METRICS:")
print(f"   Average Wasserstein Distance: {avg_wasserstein:.4f} (lower is better)")
print(f"   Mean Difference: {total_diff/10:.2f}% (lower is better)")
print(f"   KS Test Pass Rate: {sum(p > 0.05 for p in ks_pvalues)}/{len(ks_pvalues)} features")

print(f"\nCLASSIFIER ACCURACY:")
print(f"   TRTR (Train Real, Test Real):")
print(f"      Random Forest: {rf_acc_real*100:.2f}%")
print(f"      Logistic Regression: {lr_acc_real*100:.2f}%")
print(f"   TSTR (Train Synthetic, Test Real):")
print(f"      Random Forest: {rf_acc_tstr*100:.2f}% ({abs(rf_acc_real-rf_acc_tstr)*100:.2f}% gap)")
print(f"      Logistic Regression: {lr_acc_tstr*100:.2f}% ({abs(lr_acc_real-lr_acc_tstr)*100:.2f}% gap)")

print(f"\nOUTPUT FILES (in outputs/ folder):")
print(f"   1. training_loss.png - Training convergence")
print(f"   2. pca_comparison.png - PCA visualization")
print(f"   3. distributions.png - Feature distributions with WD")
print(f"   4. wasserstein_distances.png - WD per feature")
print(f"   5. accuracy_comparison.png - Classifier comparison")
print(f"   6. synthetic_data.csv - Generated data")

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)

