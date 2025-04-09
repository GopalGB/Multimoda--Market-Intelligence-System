import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path

from models.text.roberta_model import RoBERTaWrapper
from models.visual.clip_model import CLIPWrapper

class MultimodalDataset(Dataset):
    """Dataset for multimodal fusion model fine-tuning."""
    
    def __init__(
        self,
        data_file: str,
        text_column: str,
        image_column: str,
        label_column: str,
        text_model: Optional[RoBERTaWrapper] = None,
        visual_model: Optional[CLIPWrapper] = None,
        text_model_name: str = "roberta-base",
        visual_model_name: str = "openai/clip-vit-base-patch32",
        max_text_length: int = 512,
        cache_features: bool = False,
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        device: str = "cpu"
    ):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to CSV file with data
            text_column: Column name for text data
            image_column: Column name for image paths
            label_column: Column name for labels
            text_model: Pre-initialized text model (creates new one if None)
            visual_model: Pre-initialized visual model (creates new one if None)
            text_model_name: Name of text model to use if creating new one
            visual_model_name: Name of visual model to use if creating new one
            max_text_length: Maximum text length for tokenization
            cache_features: Whether to cache extracted features
            cache_dir: Directory to cache features (if enabled)
            transform: Additional transformations to apply
            device: Device to use for feature extraction
        """
        self.data = pd.read_csv(data_file)
        self.text_column = text_column
        self.image_column = image_column
        self.label_column = label_column
        self.max_text_length = max_text_length
        self.transform = transform
        self.device = device
        self.cache_features = cache_features
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Create cache directory if needed
        if self.cache_features and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize text model
        if text_model is None:
            self.text_model = RoBERTaWrapper(model_name=text_model_name, device=device)
        else:
            self.text_model = text_model
            
        # Initialize visual model
        if visual_model is None:
            self.visual_model = CLIPWrapper(model_name=visual_model_name, device=device)
        else:
            self.visual_model = visual_model
            
        # Extract sample IDs for caching
        if 'content_id' in self.data.columns:
            self.sample_ids = self.data['content_id'].values
        else:
            # Generate IDs from index if not available
            self.sample_ids = [f"sample_{i}" for i in range(len(self.data))]
            
    def __len__(self):
        return len(self.data)
    
    def _get_cache_path(self, sample_id, modality):
        """Get path for cached features."""
        return self.cache_dir / f"{sample_id}_{modality}.pt"
    
    def _load_cached_features(self, sample_id, modality):
        """Load cached features if available."""
        cache_path = self._get_cache_path(sample_id, modality)
        if cache_path.exists():
            return torch.load(cache_path, map_location=self.device)
        return None
    
    def _save_cached_features(self, sample_id, modality, features):
        """Save features to cache."""
        cache_path = self._get_cache_path(sample_id, modality)
        torch.save(features, cache_path)
    
    def __getitem__(self, idx):
        """Get dataset item with text and image features."""
        sample_id = self.sample_ids[idx]
        row = self.data.iloc[idx]
        
        # Get text features
        if self.cache_features:
            text_features = self._load_cached_features(sample_id, "text")
            if text_features is None:
                text = row[self.text_column]
                text_features = self.text_model.encode_text(text)
                self._save_cached_features(sample_id, "text", text_features)
        else:
            text = row[self.text_column]
            text_features = self.text_model.encode_text(text)
        
        # Get image features
        if self.cache_features:
            visual_features = self._load_cached_features(sample_id, "visual")
            if visual_features is None:
                image_path = row[self.image_column]
                try:
                    image = Image.open(image_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    visual_features = self.visual_model.encode_images(image)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    # Create empty visual features
                    visual_features = torch.zeros((1, self.visual_model.model.config.projection_dim), 
                                                device=self.device)
                self._save_cached_features(sample_id, "visual", visual_features)
        else:
            image_path = row[self.image_column]
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                visual_features = self.visual_model.encode_images(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Create empty visual features
                visual_features = torch.zeros((1, self.visual_model.model.config.projection_dim), 
                                            device=self.device)
        
        # Get label
        label = torch.tensor(row[self.label_column], dtype=torch.float32)
        
        # Return features and label
        return {
            'sample_id': sample_id,
            'text_features': text_features,
            'visual_features': visual_features,
            'label': label
        }

def create_data_loaders(
    config: Dict,
    text_model: Optional[RoBERTaWrapper] = None,
    visual_model: Optional[CLIPWrapper] = None,
    device: str = "cpu"
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        text_model: Pre-initialized text model
        visual_model: Pre-initialized visual model
        device: Device to use
        
    Returns:
        Dictionary with data loaders
    """
    # Create datasets
    train_dataset = MultimodalDataset(
        data_file=config['data']['train_file'],
        text_column=config['data']['text_column'],
        image_column=config['data']['image_column'],
        label_column=config['data']['label_column'],
        text_model=text_model,
        visual_model=visual_model,
        text_model_name=config['features']['text_model'],
        visual_model_name=config['features']['visual_model'],
        max_text_length=config['features']['max_text_length'],
        cache_features=config['data']['cache_features'],
        cache_dir=config['data']['cache_dir'],
        device=device
    )
    
    val_dataset = MultimodalDataset(
        data_file=config['data']['val_file'],
        text_column=config['data']['text_column'],
        image_column=config['data']['image_column'],
        label_column=config['data']['label_column'],
        text_model=text_model,
        visual_model=visual_model,
        text_model_name=config['features']['text_model'],
        visual_model_name=config['features']['visual_model'],
        max_text_length=config['features']['max_text_length'],
        cache_features=config['data']['cache_features'],
        cache_dir=config['data']['cache_dir'],
        device=device
    )
    
    test_dataset = MultimodalDataset(
        data_file=config['data']['test_file'],
        text_column=config['data']['text_column'],
        image_column=config['data']['image_column'],
        label_column=config['data']['label_column'],
        text_model=text_model,
        visual_model=visual_model,
        text_model_name=config['features']['text_model'],
        visual_model_name=config['features']['visual_model'],
        max_text_length=config['features']['max_text_length'],
        cache_features=config['data']['cache_features'],
        cache_dir=config['data']['cache_dir'],
        device=device
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    } 