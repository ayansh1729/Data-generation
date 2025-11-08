# 🎲 Tabular Data Generation with Diffusion Models

Generate high-quality synthetic tabular data using diffusion models!

## 🚀 Quick Start

### Option 1: Run the Python Script

```bash
# Make sure you're in the project directory
python examples/tabular_generation.py
```

This will:
- ✅ Load the Breast Cancer Wisconsin dataset
- ✅ Train a specialized tabular diffusion model  
- ✅ Generate synthetic data
- ✅ Create visualization plots
- ✅ Save synthetic data to CSV

### Option 2: Use in Your Code

```python
from src.models.tabular_diffusion import TabularDiffusionModel
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import torch

# Load and normalize your data
data = load_breast_cancer()
X = data.data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Create model
model = TabularDiffusionModel(
    input_dim=X.shape[1],
    hidden_dims=(256, 512, 512, 256),
    time_emb_dim=128,
    dropout=0.1
)

# Train (see examples/tabular_generation.py for full code)
# ...

# Generate synthetic data
synthetic_data = model.sample(num_samples=1000, dim=X.shape[1])
synthetic_data_denorm = scaler.inverse_transform(synthetic_data.cpu().numpy())
```

## 📊 What Gets Generated?

The script will create:

1. **`tabular_pca_comparison.png`** - PCA visualization comparing real vs synthetic data
2. **`tabular_distributions.png`** - Feature distribution comparisons
3. **`synthetic_tabular_data.csv`** - The generated synthetic dataset

## 🎯 Use Cases

### Privacy-Preserving Data Sharing
- Generate synthetic medical records without exposing patient data
- Share realistic financial data for testing
- Create training datasets that preserve privacy

### Data Augmentation
- Balance imbalanced datasets
- Increase training data for ML models
- Generate edge cases for testing

### Research & Development
- Test algorithms without real data access
- Prototype with realistic synthetic data
- Validate pipelines before deployment

## 🔧 Model Architecture

The `TabularDiffusionModel` uses:
- **MLP-based architecture** (not CNN/U-Net like image models)
- **Residual blocks** with batch normalization
- **Time embeddings** for diffusion timesteps
- **Cosine noise schedule** for smooth generation

```
Input (D dimensions)
  ↓
Time Embedding (128-dim sinusoidal)
  ↓
ResBlock (D → 256) + Time
  ↓
ResBlock (256 → 512) + Time
  ↓
ResBlock (512 → 512) + Time
  ↓
ResBlock (512 → 256) + Time
  ↓
Output (D dimensions - predicted noise)
```

## 📈 Performance

On the Breast Cancer Wisconsin dataset (569 samples, 30 features):

| Metric | Value |
|--------|-------|
| **Training time** | ~2-3 minutes (50 epochs, CPU) |
| **Parameters** | ~1.5M |
| **Mean difference** | < 5% (real vs synthetic) |
| **Std difference** | < 8% (real vs synthetic) |

## 🆚 Comparison with Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **Diffusion Models** ✅ | High quality, captures complex distributions, stable training | Slower sampling |
| GANs | Fast sampling | Training instability, mode collapse |
| VAEs | Fast, interpretable | Lower quality, blurry outputs |
| CTGAN | Good for mixed data types | Requires careful tuning |

## 🎨 Customization

### Different Datasets

```python
# Your own dataset
import pandas as pd

df = pd.read_csv('your_data.csv')
X = df.values

# Follow the same normalization and training steps
```

### Model Architecture

```python
model = TabularDiffusionModel(
    input_dim=your_features,
    hidden_dims=(128, 256, 256, 128),  # Smaller for less data
    time_emb_dim=64,                     # Can adjust
    dropout=0.2,                         # More dropout if overfitting
    use_batch_norm=True                  # False for small batches
)
```

### Training Parameters

```python
num_epochs = 100        # More epochs for better quality
batch_size = 32         # Adjust based on dataset size
learning_rate = 1e-3    # Lower if unstable
timesteps = 1000        # More steps = better quality, slower sampling
```

## 📚 Additional Resources

- **Full notebook**: `notebooks/02_tabular_data_generation.ipynb`
- **Model code**: `src/models/tabular_diffusion.py`
- **Example script**: `examples/tabular_generation.py`
- **Main README**: `../README.md`

## 🤝 Contributing

Have a use case or improvement? Open an issue or PR!

## 📄 License

MIT License - see main project LICENSE file

