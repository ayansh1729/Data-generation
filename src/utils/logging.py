"""
Logging utilities for training monitoring
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import sys


def setup_logger(
    name: str = "diffusion",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricLogger:
    """
    Logger for tracking metrics during training.
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize metric logger.
        
        Args:
            log_dir: Directory to save metrics
        """
        self.metrics = {}
        self.log_dir = Path(log_dir) if log_dir else None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, name: str, value: float, step: int):
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
        """
        if name not in self.metrics:
            self.metrics[name] = {'steps': [], 'values': []}
        
        self.metrics[name]['steps'].append(step)
        self.metrics[name]['values'].append(value)
    
    def log_dict(self, metrics: Dict[str, float], step: int):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        for name, value in metrics.items():
            self.log(name, value, step)
    
    def get_metric(self, name: str) -> Dict[str, list]:
        """
        Get logged values for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary with 'steps' and 'values'
        """
        return self.metrics.get(name, {'steps': [], 'values': []})
    
    def save(self, filename: str = 'metrics.json'):
        """
        Save metrics to file.
        
        Args:
            filename: Filename to save
        """
        if self.log_dir:
            import json
            filepath = self.log_dir / filename
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2)
    
    def load(self, filename: str = 'metrics.json'):
        """
        Load metrics from file.
        
        Args:
            filename: Filename to load
        """
        if self.log_dir:
            import json
            filepath = self.log_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    self.metrics = json.load(f)


if __name__ == "__main__":
    # Test logging utilities
    print("Testing logging utilities...")
    
    # Test logger
    logger = setup_logger(name="test", log_dir="./test_logs")
    logger.info("This is a test message")
    logger.warning("This is a warning")
    
    # Test metric logger
    metric_logger = MetricLogger(log_dir="./test_logs")
    
    for step in range(10):
        metric_logger.log('loss', 1.0 / (step + 1), step)
        metric_logger.log('accuracy', step / 10.0, step)
    
    metric_logger.save()
    print("Metrics saved")
    
    # Load and verify
    metric_logger2 = MetricLogger(log_dir="./test_logs")
    metric_logger2.load()
    loss_values = metric_logger2.get_metric('loss')
    print(f"Loaded loss values: {len(loss_values['values'])} points")
    
    print("All tests passed!")

