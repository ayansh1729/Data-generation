# 🌐 Running in Google Colab

## Quick Start - Google Colab

### Option 1: Direct Link (Easiest!)

Click this button to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayansh1729/Data-generation/blob/main/notebooks/01_getting_started.ipynb)

### Option 2: Manual Upload

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Upload notebook`
3. Choose `notebooks/01_getting_started.ipynb`

---

## 🚀 How to Use

### Step 1: Enable GPU (Important!)

1. Click `Runtime` → `Change runtime type`
2. Set `Hardware accelerator` to **GPU** (T4 or better)
3. Click `Save`

**Why?** Training is ~10x faster with GPU!

### Step 2: Run Setup Cell

The first code cell automatically:
- ✅ Clones the repository
- ✅ Installs all dependencies
- ✅ Sets up the environment
- ✅ Verifies everything works

Just click **Run** or press `Shift + Enter`!

### Step 3: Run All Cells

Click `Runtime` → `Run all` to execute the entire notebook.

**Total runtime:** ~10-15 minutes (including setup)

---

## 📊 What You'll Get

After running all cells, you'll see:

1. ✅ **Real CIFAR-10 images** - Original dataset samples
2. ✅ **Training progress** - Live progress bars showing loss
3. ✅ **Generated images** - 64 synthetic samples
4. ✅ **Comparison** - Real vs. Generated side-by-side
5. ✅ **GradCAM visualizations** - Model interpretability
6. ✅ **Attention maps** - What the model focuses on

---

## 💾 Save Your Work

### Save Generated Images
```python
# Add this cell to save images
from src.utils.visualization import save_samples_grid

save_samples_grid(
    generated_samples,
    save_dir='/content/generated_samples',
    epoch=1,
    nrow=8
)

# Download to your computer
from google.colab import files
files.download('/content/generated_samples/samples_epoch_0001.png')
```

### Save Trained Model
```python
# Add this cell to save model
model.save_checkpoint('/content/my_model.pth')

# Download to your computer
from google.colab import files
files.download('/content/my_model.pth')
```

### Save Entire Notebook
```python
# Download the notebook with outputs
from google.colab import files
files.download('/content/Data-generation/notebooks/01_getting_started.ipynb')
```

---

## ⚡ Tips for Google Colab

### 1. GPU Runtime Limits
- Free tier: ~12 hours per session
- Colab Pro: Longer sessions and better GPUs

### 2. Session Disconnections
If disconnected:
- Click `Runtime` → `Run all` to restart
- The setup cell will skip already-installed packages

### 3. Speed Up Training
```python
# Reduce epochs for faster demo
training_config = {
    'num_epochs': 1,  # Instead of 2
    'learning_rate': 2e-4,
    'gradient_clip': 1.0,
}
```

### 4. Generate Fewer Samples
```python
# Generate 16 samples instead of 64
generated_samples = model.sample(batch_size=16)
```

### 5. Monitor GPU Usage
```python
# Check GPU memory usage
!nvidia-smi
```

---

## 🔧 Troubleshooting

### "No module named 'src'"
- Re-run the first setup cell
- Make sure it completes without errors

### "CUDA out of memory"
- Reduce batch size: `batch_size=32` instead of 64
- Generate fewer samples: `batch_size=16`
- Click `Runtime` → `Restart runtime` and try again

### "Session disconnected"
- This happens after ~12 hours on free tier
- Just re-run all cells
- Your code is saved in the notebook

### Slow Training
- Make sure GPU is enabled: `Runtime` → `Change runtime type` → GPU
- Check with: `torch.cuda.is_available()` should return `True`

---

## 📱 Mobile/Tablet Users

Google Colab works on mobile! But:
- ✅ You can run and view results
- ⚠️ Editing code is harder on small screens
- 💡 Best viewed on tablet or desktop

---

## 🎯 Next Steps in Colab

Try these experiments:

### 1. Train on More Data
```python
# Change dataset
train_loader = create_dataloader(
    dataset_name='celeba',  # Faces instead of CIFAR-10
    batch_size=64,
    image_size=64
)
```

### 2. Adjust Model Size
```python
# Smaller model (faster)
model = DDPM(
    image_size=32,
    channels=3,
    timesteps=500,  # Fewer steps
    dim=32,  # Smaller dim
)
```

### 3. More Explainability
```python
# Add attention visualization
from src.explainability.attention_viz import AttentionVisualizer
visualizer = AttentionVisualizer(model)
# ... explore attention patterns
```

---

## 🌟 Share Your Results

Want to share your notebook?
1. Click `File` → `Save a copy in Drive`
2. Click `Share` and get the link
3. Post on social media with #DiffusionModels

---

## 📚 Additional Resources

- **GitHub Repo**: https://github.com/ayansh1729/Data-generation
- **Colab Docs**: https://colab.research.google.com/
- **PyTorch Docs**: https://pytorch.org/docs/

---

**Happy generating in the cloud! ☁️🎨**

