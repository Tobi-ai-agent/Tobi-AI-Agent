# src/utils/data_utils.py
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import random


class TradingChartDataset(Dataset):
    """Dataset class for trading chart images."""
    
    def __init__(self, data_dir, metadata_path=None, transform=None, target_type='price'):
        """Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing chart images
            metadata_path (str): Path to metadata CSV (optional)
            transform (callable, optional): Optional transform to apply to images
            target_type (str): Type of target ('price', 'direction', or 'custom')
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_type = target_type
        
        # Load metadata if available, otherwise scan directory
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            self.image_files = self.metadata['filename'].tolist()
            self.has_metadata = True
        else:
            # Get all image files in the directory
            self.image_files = [f for f in os.listdir(data_dir) 
                              if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.has_metadata = False
        
        # Sort image files
        self.image_files.sort()
    
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get an item from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, target)
        """
        # Get image file
        img_file = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_file)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
        
        # Get target if metadata is available
        target = torch.tensor([0.0])  # Default target
        
        if self.has_metadata:
            if self.target_type == 'price' and 'price' in self.metadata.columns:
                # Price regression
                price = self.metadata.iloc[idx]['price']
                target = torch.tensor([float(price)])
            elif self.target_type == 'direction' and 'direction' in self.metadata.columns:
                # Direction classification (0: down, 1: up)
                direction = self.metadata.iloc[idx]['direction']
                target = torch.tensor([int(direction)])
        
        return image, target


def create_dataloader(data_dir, metadata_path=None, batch_size=32, 
                     img_size=(224, 224), shuffle=True, target_type='price',
                     train_ratio=0.8):
    """Create DataLoader for training and validation.
    
    Args:
        data_dir (str): Directory containing chart images
        metadata_path (str): Path to metadata CSV (optional)
        batch_size (int): Batch size
        img_size (tuple): Image size (width, height)
        shuffle (bool): Whether to shuffle the data
        target_type (str): Type of target ('price', 'direction', or 'custom')
        train_ratio (float): Ratio of data to use for training
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    # Define transform
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = TradingChartDataset(data_dir, metadata_path, transform, target_type)
    
    # Split into train and validation sets
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader


def visualize_dataset_samples(data_dir, metadata_path=None, num_samples=5, 
                            img_size=(224, 224), save_path=None):
    """Visualize random samples from the dataset.
    
    Args:
        data_dir (str): Directory containing chart images
        metadata_path (str): Path to metadata CSV (optional)
        num_samples (int): Number of samples to visualize
        img_size (tuple): Image size (width, height)
        save_path (str): Path to save visualization (optional)
    """
    # Create dataset without normalization for visualization
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    
    dataset = TradingChartDataset(data_dir, metadata_path, transform)
    
    # Get random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Plot samples
    fig, axs = plt.subplots(1, len(indices), figsize=(15, 4))
    
    if len(indices) == 1:
        axs = [axs]
    
    for i, idx in enumerate(indices):
        img, _ = dataset[idx]
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC format for display
        
        axs[i].imshow(img)
        axs[i].axis('off')
        
        # Add metadata if available
        if dataset.has_metadata:
            metadata = dataset.metadata.iloc[idx]
            title = f"{metadata['symbol']} ({metadata['timeframe']})"
            if 'price' in metadata:
                title += f"\nPrice: ${metadata['price']:.2f}"
            axs[i].set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()