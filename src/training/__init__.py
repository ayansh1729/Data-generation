"""
Training package initialization
"""

# from src.training.trainer import DiffusionTrainer  # TODO: Implement trainer
from src.training.losses import get_loss_function
from src.training.scheduler import get_scheduler

__all__ = [
    # "DiffusionTrainer",
    "get_loss_function",
    "get_scheduler",
]

