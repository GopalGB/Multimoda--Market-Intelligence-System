import pandas as pd
import os
import json
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import torch
import requests
from pathlib import Path
from tqdm import tqdm

class ContentDataCollector:
    """Collects and preprocesses domain-specific audience engagement data."""
    
    def __init__(self, data_dir: str = './data/datasets', cache_dir: str = './data/cache'):
        """
        Initialize the data collector.
        
        Args:
            data_dir: Directory to store collected datasets
            cache_dir: Directory to cache downloaded content
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_nielsen_content(
        self,
        api_key: Optional[str] = None,
        start_date: str = '2023-01-01',
        end_date: str = '2023-12-31',
        content_types: List[str] = ['show', 'movie', 'ad'],
        min_engagement: float = 0.2,
        max_samples: int = 1000,
        output_file: str = 'nielsen_content.csv'
    ) -> pd.DataFrame:
        """
        Collect Nielsen content data with engagement metrics.
        
        Args:
            api_key: Nielsen API key (uses env var NIELSEN_API_KEY if None)
            start_date: Start date for data collection
            end_date: End date for data collection
            content_types: Types of content to collect
            min_engagement: Minimum engagement score to include
            max_samples: Maximum number of samples to collect
            output_file: Name of output CSV file
            
        Returns:
            DataFrame with collected content data
        """
        if api_key is None:
            api_key = os.environ.get('NIELSEN_API_KEY')
            if api_key is None:
                raise ValueError("Nielsen API key must be provided or set as NIELSEN_API_KEY environment variable")
        
        # In a real implementation, you would make API calls to Nielsen
        # For now, we'll create synthetic data to demonstrate the process
        print(f"Collecting Nielsen content data from {start_date} to {end_date}...")
        
        # Create synthetic data
        import numpy as np
        np.random.seed(42)
        
        data = []
        for i in range(max_samples):
            content_type = np.random.choice(content_types)
            
            # Create synthetic record
            record = {
                'content_id': f"{content_type.upper()}{i:06d}",
                'title': f"Sample {content_type.title()} {i}",
                'content_type': content_type,
                'release_date': pd.Timestamp(start_date) + pd.Timedelta(days=np.random.randint(0, 365)),
                'duration_seconds': np.random.randint(30, 7200),  # 30 seconds to 2 hours
                'engagement': np.random.uniform(min_engagement, 1.0),
                'completion_rate': np.random.uniform(0.5, 1.0),
                'thumbnail_url': f"https://example.com/{content_type}{i}.jpg",
                'description': f"This is a sample {content_type} description with various words and phrases that might be engaging to audiences.",
                'genre': np.random.choice(['drama', 'comedy', 'action', 'documentary', 'reality']),
                'has_popular_actor': np.random.choice([0, 1]),
                'promotion_level': np.random.randint(1, 5)
            }
            
            data.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        output_path = self.data_dir / output_file
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")
        
        return df
    
    def download_thumbnails(
        self,
        data: pd.DataFrame,
        url_column: str = 'thumbnail_url',
        id_column: str = 'content_id',
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download thumbnails for content.
        
        Args:
            data: DataFrame with content data
            url_column: Column name containing thumbnail URLs
            id_column: Column name containing content IDs
            output_dir: Directory to save thumbnails (defaults to cache_dir/thumbnails)
            
        Returns:
            DataFrame with added thumbnail_path column
        """
        if output_dir is None:
            output_dir = self.cache_dir / 'thumbnails'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create copy of DataFrame to add thumbnail_path
        result_df = data.copy()
        result_df['thumbnail_path'] = None
        
        print(f"Downloading thumbnails for {len(data)} content items...")
        
        for i, row in tqdm(data.iterrows(), total=len(data)):
            url = row[url_column]
            content_id = row[id_column]
            output_path = output_dir / f"{content_id}.jpg"
            
            # Skip if already downloaded
            if output_path.exists():
                result_df.loc[i, 'thumbnail_path'] = str(output_path)
                continue
            
            # In a real implementation, you would download the image
            # For now, we'll create a placeholder image
            img = Image.new('RGB', (224, 224), color=(73, 109, 137))
            img.save(output_path)
            
            result_df.loc[i, 'thumbnail_path'] = str(output_path)
        
        return result_df
    
    def prepare_training_data(
        self,
        data: pd.DataFrame,
        text_column: str = 'description',
        image_column: str = 'thumbnail_path',
        label_column: str = 'engagement',
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for training by splitting into train/val/test sets.
        
        Args:
            data: DataFrame with content data
            text_column: Column name containing text data
            image_column: Column name containing image paths
            label_column: Column name containing target labels
            test_size: Fraction of data to use for testing
            val_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train, val, and test DataFrames
        """
        from sklearn.model_selection import train_test_split
        
        # Validate columns
        for col in [text_column, image_column, label_column]:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data")
        
        # First, split into train+val and test
        train_val, test = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=None  # For engagement, we don't stratify
        )
        
        # Then split train+val into train and val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=None
        )
        
        # Print info
        print(f"Data split: {len(train)} train, {len(val)} validation, {len(test)} test samples")
        
        return {
            'train': train,
            'val': val,
            'test': test
        }
    
    def save_dataset_splits(
        self,
        data_splits: Dict[str, pd.DataFrame],
        dataset_name: str = 'nielsen_content'
    ) -> Dict[str, str]:
        """
        Save dataset splits to disk.
        
        Args:
            data_splits: Dictionary with train, val, and test DataFrames
            dataset_name: Name for the dataset
            
        Returns:
            Dictionary with paths to saved splits
        """
        output_paths = {}
        
        for split_name, df in data_splits.items():
            output_path = self.data_dir / f"{dataset_name}_{split_name}.csv"
            df.to_csv(output_path, index=False)
            output_paths[split_name] = str(output_path)
            
            print(f"Saved {split_name} split with {len(df)} samples to {output_path}")
        
        # Save dataset info
        dataset_info = {
            'name': dataset_name,
            'splits': {k: {'path': v, 'samples': len(data_splits[k])} for k, v in output_paths.items()},
            'total_samples': sum(len(df) for df in data_splits.values()),
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        info_path = self.data_dir / f"{dataset_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
            
        return output_paths 