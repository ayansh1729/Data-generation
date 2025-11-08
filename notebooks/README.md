# Quick Start Notebook Guide

## Creating a Jupyter Notebook

To create the Getting Started notebook, run:

```bash
jupyter notebook
```

Then create a new notebook with the following cells:

### Cell 1 (Markdown)
```markdown
# Synthetic Data Generation with Diffusion Models

Quick start guide for the framework.
```

### Cell 2 (Code)
```python
import torch
from src.models.diffusion import DDPM
from src.data.dataset import create_dataloader
from src.utils.visualization import plot_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### Cell 3 (Code) - Load Data
```python
train_loader = create_dataloader(
    dataset_name='cifar10',
    batch_size=64,
    image_size=32
)

real_batch = next(iter(train_loader))
plot_samples(real_batch, nrow=8, title="Real Samples")
```

### Cell 4 (Code) - Create Model
```python
model = DDPM(
    image_size=32,
    channels=3,
    timesteps=1000,
    noise_schedule='cosine'
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Cell 5 (Code) - Train (Simplified)
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(2):
    for batch in train_loader:
        batch = batch.to(device)
        outputs = model(batch)
        loss = outputs['loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
        break  # One batch for demo
```

### Cell 6 (Code) - Generate
```python
model.eval()
with torch.no_grad():
    samples = model.sample(batch_size=16)

plot_samples(samples, nrow=4, title="Generated Samples")
```

### Cell 7 (Code) - Explain
```python
from src.explainability.gradcam import GradCAMExplainer

explainer = GradCAMExplainer(model)
x = real_batch[:4].to(device)
t = torch.randint(0, 1000, (4,)).to(device)

gradcams = explainer.compute_gradcam(x, t)
visualizations = explainer.visualize_gradcam(x, t)

print(f"Computed GradCAM for {len(gradcams)} layers")
```

## Alternative: Run as Script

Save this as `quickstart.py`:

```python
import torch
from src.models.diffusion import DDPM
from src.data.dataset import create_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader = create_dataloader('cifar10', batch_size=64, image_size=32)

# Create model
model = DDPM(image_size=32, channels=3, timesteps=1000).to(device)

# Quick training
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
for batch in train_loader:
    batch = batch.to(device)
    outputs = model(batch)
    loss = outputs['loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f}")
    break

# Generate
samples = model.sample(batch_size=16)
print(f"Generated {samples.shape[0]} samples")
```

Run with:
```bash
python quickstart.py
```

