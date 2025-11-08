"""
Learning rate schedulers for diffusion model training
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR, ExponentialLR
import math
from typing import Optional


class WarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup.
    
    Args:
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        base_scheduler: Base scheduler to use after warmup
        last_epoch: Last epoch number
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Use base scheduler if provided
            if self.base_scheduler is not None:
                return self.base_scheduler.get_last_lr()
            return self.base_lrs
    
    def step(self, epoch=None):
        super().step(epoch)
        if self.base_scheduler is not None and self.last_epoch >= self.warmup_steps:
            self.base_scheduler.step()


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts.
    
    Args:
        optimizer: Optimizer
        first_cycle_steps: Number of steps in first cycle
        cycle_mult: Multiplier for cycle length after each restart
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        warmup_steps: Number of warmup steps
        gamma: Decay factor for max_lr after each cycle
        last_epoch: Last epoch number
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        # Initialize learning rate
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) *
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) /
                                  (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Linear warmup followed by cosine annealing.
    
    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Maximum number of epochs
        warmup_start_lr: Starting learning rate for warmup
        eta_min: Minimum learning rate
        last_epoch: Last epoch number
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'cosine',
    **kwargs
) -> _LRScheduler:
    """
    Get learning rate scheduler by type.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler
        **kwargs: Additional arguments
        
    Returns:
        Scheduler instance
    """
    if scheduler_type == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0.0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'warmup':
        warmup_steps = kwargs.get('warmup_steps', 1000)
        base_scheduler = kwargs.get('base_scheduler', None)
        return WarmupScheduler(optimizer, warmup_steps, base_scheduler)
    
    elif scheduler_type == 'cosine_warmup':
        warmup_epochs = kwargs.get('warmup_epochs', 10)
        max_epochs = kwargs.get('max_epochs', 100)
        warmup_start_lr = kwargs.get('warmup_start_lr', 0.0)
        eta_min = kwargs.get('eta_min', 0.0)
        return LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs,
            max_epochs,
            warmup_start_lr,
            eta_min
        )
    
    elif scheduler_type == 'cosine_restarts':
        first_cycle_steps = kwargs.get('first_cycle_steps', 100)
        cycle_mult = kwargs.get('cycle_mult', 1.0)
        max_lr = kwargs.get('max_lr', 0.1)
        min_lr = kwargs.get('min_lr', 0.001)
        warmup_steps = kwargs.get('warmup_steps', 0)
        gamma = kwargs.get('gamma', 1.0)
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps,
            cycle_mult,
            max_lr,
            min_lr,
            warmup_steps,
            gamma
        )
    
    elif scheduler_type == 'none' or scheduler_type is None:
        # No scheduler
        return None
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    # Test schedulers
    print("Testing learning rate schedulers...")
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test cosine annealing with warmup
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine_warmup',
        warmup_epochs=10,
        max_epochs=100
    )
    
    lrs = []
    for epoch in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    print(f"Learning rate schedule (first 10): {lrs[:10]}")
    print(f"Learning rate schedule (last 10): {lrs[-10:]}")
    
    # Test warmup scheduler
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)
    base_scheduler = CosineAnnealingLR(optimizer2, T_max=90)
    scheduler2 = WarmupScheduler(optimizer2, warmup_steps=10, base_scheduler=base_scheduler)
    
    lrs2 = []
    for epoch in range(100):
        lrs2.append(optimizer2.param_groups[0]['lr'])
        scheduler2.step()
    
    print(f"Warmup schedule (first 10): {lrs2[:10]}")
    print(f"Warmup schedule (last 10): {lrs2[-10:]}")
    
    print("All tests passed!")

