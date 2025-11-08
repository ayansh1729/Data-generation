"""
Configuration management utilities
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf, DictConfig
import json


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to OmegaConf for dot notation access
    config = OmegaConf.create(config_dict)
    
    return config


def save_config(config: Union[Dict, DictConfig], save_path: Union[str, Path]):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dict or DictConfig
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: DictConfig, override_config: Dict) -> DictConfig:
    """
    Merge override config into base config.
    
    Args:
        base_config: Base configuration
        override_config: Override values
        
    Returns:
        Merged configuration
    """
    merged = OmegaConf.merge(base_config, override_config)
    return merged


def config_to_dict(config: DictConfig) -> Dict:
    """
    Convert OmegaConf config to regular dict.
    
    Args:
        config: OmegaConf configuration
        
    Returns:
        Regular Python dictionary
    """
    return OmegaConf.to_container(config, resolve=True)


def dict_to_config(d: Dict) -> DictConfig:
    """
    Convert dict to OmegaConf config.
    
    Args:
        d: Python dictionary
        
    Returns:
        OmegaConf configuration
    """
    return OmegaConf.create(d)


def save_config_json(config: Union[Dict, DictConfig], save_path: Union[str, Path]):
    """
    Save configuration as JSON.
    
    Args:
        config: Configuration
        save_path: Path to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config_json(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration from JSON.
    
    Args:
        config_path: Path to JSON config
        
    Returns:
        Configuration object
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return OmegaConf.create(config_dict)


class ConfigManager:
    """
    Manage experiment configurations.
    """
    
    def __init__(self, base_config_path: Optional[Union[str, Path]] = None):
        """
        Initialize config manager.
        
        Args:
            base_config_path: Optional base config path
        """
        if base_config_path:
            self.config = load_config(base_config_path)
        else:
            self.config = OmegaConf.create({})
    
    def update(self, **kwargs):
        """Update config with keyword arguments."""
        self.config = OmegaConf.merge(self.config, kwargs)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by key.
        
        Args:
            key: Dot-separated key (e.g., 'model.image_size')
            default: Default value if key not found
            
        Returns:
            Config value
        """
        try:
            return OmegaConf.select(self.config, key, default=default)
        except:
            return default
    
    def set(self, key: str, value: Any):
        """
        Set config value by key.
        
        Args:
            key: Dot-separated key
            value: Value to set
        """
        OmegaConf.update(self.config, key, value)
    
    def save(self, save_path: Union[str, Path]):
        """Save configuration."""
        save_config(self.config, save_path)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return config_to_dict(self.config)


if __name__ == "__main__":
    # Test configuration utilities
    print("Testing configuration utilities...")
    
    # Create test config
    test_config = {
        'model': {
            'type': 'DDPM',
            'image_size': 64
        },
        'training': {
            'batch_size': 128,
            'learning_rate': 0.0002
        }
    }
    
    # Save and load
    save_config(test_config, 'test_config.yaml')
    loaded = load_config('test_config.yaml')
    
    print(f"Loaded config: {loaded}")
    print(f"Image size: {loaded.model.image_size}")
    
    # Test ConfigManager
    manager = ConfigManager()
    manager.update(**test_config)
    print(f"Manager config: {manager.get('model.type')}")
    
    print("All tests passed!")

