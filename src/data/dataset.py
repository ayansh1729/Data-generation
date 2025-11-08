"""
Dataset classes for diffusion models

Supports various data types: images, tabular data, time series
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple, Union, List
from PIL import Image
import pandas as pd
from torchvision import transforms, datasets


class DiffusionDataset(Dataset):
    """
    Generic dataset for diffusion models.
    
    Args:
        data_path: Path to the dataset
        transform: Optional transform to apply
        split: Dataset split ('train', 'val', 'test')
    """
    
    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        split: str = 'train'
    ):
        self.data_path = Path(data_path) if data_path else None
        self.transform = transform
        self.split = split
        self.data = None
        
        if data_path:
            self._load_data()
    
    def _load_data(self):
        """Load data from path."""
        if self.data_path.is_dir():
            # Assume image directory
            self.data = self._load_image_directory()
        elif self.data_path.suffix in ['.csv', '.pkl', '.parquet']:
            # Tabular data
            self.data = self._load_tabular_data()
        else:
            raise ValueError(f"Unsupported data format: {self.data_path.suffix}")
    
    def _load_image_directory(self) -> List[Path]:
        """Load image paths from directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(self.data_path.glob(f'**/*{ext}'))
        
        return sorted(image_paths)
    
    def _load_tabular_data(self) -> pd.DataFrame:
        """Load tabular data."""
        if self.data_path.suffix == '.csv':
            return pd.read_csv(self.data_path)
        elif self.data_path.suffix == '.pkl':
            return pd.read_pickle(self.data_path)
        elif self.data_path.suffix == '.parquet':
            return pd.read_parquet(self.data_path)
    
    def __len__(self) -> int:
        if self.data is None:
            return 0
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if isinstance(self.data, list):
            # Image data
            img_path = self.data[idx]
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img
        
        elif isinstance(self.data, pd.DataFrame):
            # Tabular data
            row = self.data.iloc[idx].values.astype(np.float32)
            
            if self.transform:
                row = self.transform(row)
            
            return torch.from_numpy(row)
        
        else:
            raise ValueError("No data loaded")


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset wrapper."""
    
    def __init__(self, root: str = './data/raw', train: bool = True, transform: Optional[Callable] = None):
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.dataset[idx]  # Ignore labels for unsupervised learning
        return img


class CelebADataset(Dataset):
    """CelebA dataset wrapper."""
    
    def __init__(self, root: str = './data/raw', split: str = 'train', transform: Optional[Callable] = None):
        self.dataset = datasets.CelebA(
            root=root,
            split=split,
            download=True,
            transform=transform
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.dataset[idx]
        return img


class ImageFolderDataset(Dataset):
    """Custom image folder dataset."""
    
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.dataset[idx]
        return img


class TabularDataset(Dataset):
    """Dataset for tabular data."""
    
    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame, str, Path],
        normalize: bool = True,
        transform: Optional[Callable] = None
    ):
        if isinstance(data, (str, Path)):
            data = pd.read_csv(data)
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.data = data.astype(np.float32)
        self.transform = transform
        
        if normalize:
            self.mean = self.data.mean(axis=0)
            self.std = self.data.std(axis=0) + 1e-8
            self.data = (self.data - self.mean) / self.std
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return torch.from_numpy(sample)


class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""
    
    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        sequence_length: int = 100,
        stride: int = 1,
        transform: Optional[Callable] = None
    ):
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.data = data.astype(np.float32)
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[np.ndarray]:
        """Create sliding window sequences."""
        sequences = []
        for i in range(0, len(self.data) - self.sequence_length + 1, self.stride):
            seq = self.data[i:i + self.sequence_length]
            sequences.append(seq)
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]
        
        if self.transform:
            seq = self.transform(seq)
        
        return torch.from_numpy(seq)


def create_dataloader(
    dataset_name: str = 'cifar10',
    data_path: Optional[str] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 64,
    split: str = 'train',
    **kwargs
) -> DataLoader:
    """
    Create a dataloader for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'celeba', 'custom', etc.)
        data_path: Path to custom dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        image_size: Target image size
        split: Dataset split
        **kwargs: Additional arguments
        
    Returns:
        DataLoader instance
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip() if split == 'train' else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    # Create dataset
    if dataset_name.lower() == 'cifar10':
        dataset = CIFAR10Dataset(
            root=data_path or './data/raw',
            train=(split == 'train'),
            transform=transform
        )
    
    elif dataset_name.lower() == 'celeba':
        dataset = CelebADataset(
            root=data_path or './data/raw',
            split=split,
            transform=transform
        )
    
    elif dataset_name.lower() == 'custom' and data_path:
        dataset = DiffusionDataset(
            data_path=data_path,
            transform=transform,
            split=split
        )
    
    elif dataset_name.lower() == 'imagefolder' and data_path:
        dataset = ImageFolderDataset(
            root=data_path,
            transform=transform
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == 'train')
    )
    
    return dataloader


if __name__ == "__main__":
    # Test datasets
    print("Testing CIFAR-10 dataset...")
    dataloader = create_dataloader(
        dataset_name='cifar10',
        batch_size=16,
        num_workers=0,
        image_size=64
    )
    
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")
    print(f"Batch range: [{batch.min():.2f}, {batch.max():.2f}]")
    
    # Test tabular dataset
    print("\nTesting tabular dataset...")
    data = np.random.randn(1000, 10)
    tabular_dataset = TabularDataset(data)
    print(f"Tabular dataset size: {len(tabular_dataset)}")
    sample = tabular_dataset[0]
    print(f"Sample shape: {sample.shape}")

