"""
Synthetic Data Generation Framework with Diffusion Models and Explainability
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.models.diffusion import DDPM
from src.models.unet import UNet
from src.explainability.gradcam import GradCAMExplainer
# from src.training.trainer import DiffusionTrainer  # TODO: Implement trainer

__all__ = [
    "DDPM",
    "UNet",
    "GradCAMExplainer",
    # "DiffusionTrainer",
]

