"""
Dataset utilities for oil spill detection from SAR imagery.
Supports Sentinel-1 and PALSAR satellite data.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional, Dict


class OilSpillDataset(Dataset):
    """
    Dataset class for oil spill segmentation.
    Loads SAR images and corresponding binary segmentation masks.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        sensor: str = 'sentinel',
        transform: Optional[A.Compose] = None,
        return_path: bool = False
    ):
        """
        Args:
            root_dir: Path to dataset root
            split: 'train' or 'test'
            sensor: 'sentinel', 'palsar', or 'both'
            transform: Albumentations transform pipeline
            return_path: Whether to return image path
        """
        self.root_dir = root_dir
        self.split = split
        self.sensor = sensor
        self.transform = transform
        self.return_path = return_path

        self.image_paths = []
        self.label_paths = []

        if sensor in ['sentinel', 'both']:
            self._load_sensor_data('sentinel')
        if sensor in ['palsar', 'both']:
            self._load_sensor_data('palsar')

    def _load_sensor_data(self, sensor_name: str):
        """Load image and label paths for a specific sensor."""
        image_dir = os.path.join(self.root_dir, self.split, sensor_name, 'image')
        label_dir = os.path.join(self.root_dir, self.split, sensor_name, 'label')

        for img_name in sorted(os.listdir(image_dir)):
            img_path = os.path.join(image_dir, img_name)
            label_path = os.path.join(label_dir, img_name)

            if os.path.exists(label_path):
                self.image_paths.append(img_path)
                self.label_paths.append(label_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image and label
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        label = np.array(Image.open(self.label_paths[idx]).convert('L'))

        # Binarize label (0 or 1)
        label = (label > 127).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        else:
            # Default: normalize and convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            label = torch.from_numpy(label).float()

        if self.return_path:
            return image, label.unsqueeze(0), self.image_paths[idx]

        return image, label.unsqueeze(0)


def get_transforms(split: str = 'train', img_size: int = 256) -> A.Compose:
    """
    Get augmentation transforms for training/validation.

    Args:
        split: 'train' or 'test'
        img_size: Target image size

    Returns:
        Albumentations transform pipeline
    """
    if split == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_dataloaders(
    root_dir: str,
    sensor: str = 'sentinel',
    batch_size: int = 16,
    img_size: int = 256,
    num_workers: int = 4,
    val_split: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        root_dir: Path to dataset root
        sensor: 'sentinel', 'palsar', or 'both'
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of workers
        val_split: Validation split ratio from training data

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = get_transforms('train', img_size)
    test_transform = get_transforms('test', img_size)

    # Full training dataset
    full_train_dataset = OilSpillDataset(
        root_dir, split='train', sensor=sensor, transform=train_transform
    )

    # Split into train and validation
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Override transform for validation (no augmentation)
    val_dataset_clean = OilSpillDataset(
        root_dir, split='train', sensor=sensor, transform=test_transform
    )
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset_clean, val_indices)

    # Test dataset
    test_dataset = OilSpillDataset(
        root_dir, split='test', sensor=sensor, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_class_weights(dataset: OilSpillDataset) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance.

    Args:
        dataset: OilSpillDataset instance

    Returns:
        Tensor of class weights [background_weight, oil_weight]
    """
    total_pixels = 0
    oil_pixels = 0

    for i in range(min(len(dataset), 500)):  # Sample for efficiency
        _, label = dataset[i]
        total_pixels += label.numel()
        oil_pixels += label.sum().item()

    bg_pixels = total_pixels - oil_pixels

    # Inverse frequency weighting
    weight_bg = total_pixels / (2 * bg_pixels + 1e-6)
    weight_oil = total_pixels / (2 * oil_pixels + 1e-6)

    return torch.tensor([weight_bg, weight_oil])


if __name__ == "__main__":
    # Test dataset loading
    root = '/teamspace/studios/this_studio/cv-project/dataset'

    print("Testing Sentinel dataset...")
    dataset = OilSpillDataset(root, split='train', sensor='sentinel')
    print(f"  Train samples: {len(dataset)}")
    img, label = dataset[0]
    print(f"  Image shape: {img.shape}, Label shape: {label.shape}")

    print("\nTesting PALSAR dataset...")
    dataset = OilSpillDataset(root, split='train', sensor='palsar')
    print(f"  Train samples: {len(dataset)}")

    print("\nTesting combined dataset...")
    dataset = OilSpillDataset(root, split='train', sensor='both')
    print(f"  Train samples: {len(dataset)}")

    print("\nTesting dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(root, sensor='sentinel', batch_size=8)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
