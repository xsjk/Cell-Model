import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Callable
import numpy as np

from .dataset import BaseDataset, LoadingMode


class VoxelDataset(Dataset):
    def __init__(self, dataset: BaseDataset, normalize: bool = True, transform: Callable | None = None):
        self.dataset = dataset
        self.normalize = normalize
        self.dataset_name = dataset.dataset_name
        self.transform = transform

        # Build sample index mapping
        self.sample_index = []
        for data_idx in range(len(self.dataset)):
            image, mask = self.dataset[data_idx]
            sample_count = self._get_sample_count(image, mask)
            for sample_idx in range(sample_count):
                self.sample_index.append((data_idx, sample_idx))

    def _get_sample_count(self, image: np.ndarray, mask: np.ndarray) -> int:
        match image.ndim:
            case 5:  # (C, T, Z, Y, X)
                return image.shape[1]  # Time dimension
            case 4:  # (C, Z, Y, X)
                return 1
            case 3:  # (Z, Y, X)
                return 1
            case _:
                raise ValueError(f"Unsupported image dimensions: {image.shape}")

    def _extract_sample(self, image: np.ndarray, mask: np.ndarray, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
        match image.ndim:
            case 5:  # (C, T, Z, Y, X)
                return image[:, sample_idx, :, :, :], mask[:, sample_idx, :, :, :]
            case 4:  # (C, Z, Y, X)
                return image, mask
            case 3:  # (Z, Y, X)
                return image[np.newaxis, ...], mask[np.newaxis, ...]
            case _:
                raise ValueError(f"Unsupported image dimensions: {image.shape}")

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_idx, sample_idx = self.sample_index[idx]
        image, mask = self.dataset[data_idx]

        # Extract sample
        image_sample, mask_sample = self._extract_sample(image, mask, sample_idx)

        # Convert to tensors
        image_tensor = torch.from_numpy(image_sample).float()
        mask_tensor = torch.from_numpy(mask_sample).float()

        # Normalize if requested
        if self.normalize and image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
        return image_tensor, mask_tensor


def create_voxel_dataloader(dataset_class: type, loading_mode: LoadingMode = LoadingMode.ON_DEMAND, config_path: str | None = None, batch_size: int = 1, shuffle: bool = True, num_workers: int = 0, transform: Callable | None = None, normalize: bool = True) -> DataLoader:
    base_dataset = dataset_class(loading_mode=loading_mode, config_path=config_path)
    torch_dataset = VoxelDataset(dataset=base_dataset, normalize=normalize, transform=transform)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def create_train_val_voxel_dataloaders(
    dataset_class: type,
    train_split: float = 0.8,
    loading_mode: LoadingMode = LoadingMode.ON_DEMAND,
    config_path: str | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    transform: Callable | None = None,
    normalize: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    base_dataset = dataset_class(loading_mode=loading_mode, config_path=config_path)
    torch_dataset = VoxelDataset(dataset=base_dataset, normalize=normalize, transform=transform)

    # Split dataset
    total_size = len(torch_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(torch_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
