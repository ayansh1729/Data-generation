# Getting Started with Synthetic Data Generation Framework

## Installation

### Step 1: Clone and Navigate
```bash
cd Data-Generation-Project
```

### Step 2: Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Quick Test

### Test Individual Components

```python
# Test U-Net
python src/models/unet.py

# Test Diffusion Model
python src/models/diffusion.py

# Test Dataset
python src/data/dataset.py

# Test Losses
python src/training/losses.py
```

## Train Your First Model

### 1. Using CIFAR-10
```python
from src.models.diffusion import DDPM
from src.data.dataset import create_dataloader
from src.training.losses import get_loss_function
import torch

# Create dataloader
dataloader = create_dataloader(
    dataset_name='cifar10',
    batch_size=64,
    image_size=32
)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DDPM(
    image_size=32,
    channels=3,
    timesteps=1000,
    noise_schedule='cosine'
).to(device)

# Training loop (simplified)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
loss_fn = get_loss_function('mse')

for epoch in range(10):
    for batch in dataloader:
        batch = batch.to(device)
        
        # Forward pass
        outputs = model(batch)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
```

### 2. Generate Samples
```python
# After training
samples = model.sample(batch_size=16)

# Visualize
import matplotlib.pyplot as plt
import torchvision.utils as vutils

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(vutils.make_grid(samples, padding=2, normalize=True).cpu().permute(1, 2, 0))
plt.show()
```

### 3. Explain Generations
```python
from src.explainability.gradcam import GradCAMExplainer

# Create explainer
explainer = GradCAMExplainer(model)

# Generate with explanations
x = torch.randn(4, 3, 32, 32).to(device)
t = torch.randint(0, 1000, (4,)).to(device)

gradcams = explainer.compute_gradcam(x, t)
visualizations = explainer.visualize_gradcam(x, t)

print(f"Computed GradCAM for {len(gradcams)} layers")
```

## Next Steps

1. **Explore Notebooks**: Check `notebooks/` for detailed tutorials
2. **Customize Config**: Modify `configs/base_config.yaml` for your needs
3. **Try Different Datasets**: Use CelebA, custom images, or tabular data
4. **Experiment with Explainability**: Try different interpretation methods

## Common Issues

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `image_size`
- Use gradient checkpointing

### Slow Training
- Enable mixed precision: Set `system.mixed_precision: true`
- Increase `num_workers` in dataloader
- Use smaller model (reduce `dim` in U-Net)

### Import Errors
- Make sure you ran `pip install -e .`
- Check virtual environment is activated

## Resources

- **Paper**: Denoising Diffusion Probabilistic Models
- **Code Examples**: See `notebooks/`
- **Configuration**: See `configs/base_config.yaml`
- **API Docs**: Coming soon

## Support

- GitHub Issues: For bug reports
- Discussions: For questions and ideas
- Email: your.email@example.com

