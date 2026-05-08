"""
Tamil Handwritten Character Dataset Loader
Loads images from class-based directory structure for training
"""

import os
import glob
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from char_mapping import get_char_mapping, class_id_to_char


class TamilCharDataset(Dataset):
    """
    Dataset for Tamil handwritten characters organized in class-based directories.
    
    Directory structure:
        root/
            0/
                image1.bmp
                image2.bmp
                ...
            1/
                image1.bmp
                ...
            ...
            155/
                ...
    
    Args:
        root_dir: Root directory containing class subdirectories
        transform: Optional transform to be applied on images
        target_transform: Optional transform to be applied on labels
        subset_size: If specified, only use this many samples (for quick testing)
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        subset_size: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.char_mapping = get_char_mapping()
        
        # Load all image paths and their corresponding class labels
        self.samples = []
        self._load_samples()
        
        # Apply subset if specified
        if subset_size is not None and subset_size < len(self.samples):
            self.samples = self.samples[:subset_size]
        
        print(f"Loaded {len(self.samples)} samples from {root_dir}")
    
    def _load_samples(self):
        """Load all image paths and their class labels."""
        # Get all class directories
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        for class_dir in class_dirs:
            try:
                class_id = int(class_dir.name)
                if class_id not in self.char_mapping:
                    print(f"Warning: Class ID {class_id} not in character mapping, skipping")
                    continue
                
                # Get all .bmp files in this class directory
                image_files = list(class_dir.glob('*.bmp'))
                
                for img_path in image_files:
                    self.samples.append((str(img_path), class_id))
                    
            except ValueError:
                print(f"Warning: Invalid class directory name: {class_dir.name}")
                continue
        
        print(f"Found {len(self.samples)} images across {len(class_dirs)} classes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            image: Transformed image tensor
            label: Tamil character string
        """
        img_path, class_id = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get Tamil character label
        label = class_id_to_char(class_id)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """Returns the distribution of samples across classes."""
        distribution = {}
        for _, class_id in self.samples:
            char = class_id_to_char(class_id)
            distribution[char] = distribution.get(char, 0) + 1
        return distribution


def get_default_transforms(img_size=(32, 128), augment=False):
    """
    Get default image transforms for Tamil character recognition.
    
    Args:
        img_size: Target image size (height, width)
        augment: Whether to apply data augmentation
    
    Returns:
        torchvision.transforms.Compose object
    """
    transform_list = []
    
    if augment:
        # Data augmentation for training
        transform_list.extend([
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    
    # Common transforms
    transform_list.extend([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (32, 128),
    subset_size: Optional[int] = None
):
    """
    Create training and testing dataloaders.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        img_size: Target image size (height, width)
        subset_size: If specified, use only this many samples for quick testing
    
    Returns:
        train_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = TamilCharDataset(
        train_dir,
        transform=get_default_transforms(img_size, augment=True),
        subset_size=subset_size
    )
    
    test_dataset = TamilCharDataset(
        test_dir,
        transform=get_default_transforms(img_size, augment=False),
        subset_size=subset_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test the dataset loader
    import sys
    
    # Test with a small subset
    train_dir = r"c:\D\Projects\image to text\own model\archive\train"
    
    print("Testing TamilCharDataset...")
    print("-" * 50)
    
    # Create dataset with transforms
    transform = get_default_transforms(img_size=(32, 128), augment=False)
    dataset = TamilCharDataset(train_dir, transform=transform, subset_size=100)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {img.shape}")
        print(f"  Image dtype: {img.dtype}")
        print(f"  Label: '{label}'")
        print(f"  Image value range: [{img.min():.3f}, {img.max():.3f}]")
        
        # Show class distribution
        print("\nClass distribution (first 10):")
        dist = dataset.get_class_distribution()
        for i, (char, count) in enumerate(sorted(dist.items(), key=lambda x: x[1], reverse=True)[:10]):
            print(f"  {char}: {count} samples")
    
    print("\n" + "-" * 50)
    print("Dataset test completed successfully!")
