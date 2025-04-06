# data/preprocessing/image_preprocessor.py
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
from torchvision import transforms
import io
import base64
import requests
from urllib.parse import urlparse

class ImagePreprocessor:
    """
    Image preprocessor for media content analysis.
    Handles loading, resizing, normalization, and feature extraction.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values
            augment: Whether to use data augmentation
        """
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment
        
        # Define image transformations
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if normalize else transforms.Lambda(lambda x: x)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if normalize else transforms.Lambda(lambda x: x)
            ])
    
    def load_image(
        self,
        image_input: Union[str, bytes, np.ndarray, Image.Image]
    ) -> Image.Image:
        """
        Load an image from various input types.
        
        Args:
            image_input: Image input (file path, URL, bytes, numpy array, or PIL Image)
            
        Returns:
            PIL Image
        """
        if isinstance(image_input, str):
            # Check if it's a URL
            if urlparse(image_input).scheme in ('http', 'https'):
                response = requests.get(image_input, stream=True)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            else:
                # Assume it's a file path
                image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # Convert from OpenCV BGR to RGB
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_input.astype('uint8')).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        return image
    
    def preprocess(
        self,
        image_input: Union[str, bytes, np.ndarray, Image.Image]
    ) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image_input: Image input (file path, URL, bytes, numpy array, or PIL Image)
            
        Returns:
            Preprocessed tensor [C, H, W]
        """
        # Load the image
        image = self.load_image(image_input)
        
        # Apply transformations
        tensor = self.transform(image)
        
        return tensor
    
    def preprocess_batch(
        self,
        image_inputs: List[Union[str, bytes, np.ndarray, Image.Image]]
    ) -> torch.Tensor:
        """
        Preprocess a batch of images for model input.
        
        Args:
            image_inputs: List of image inputs
            
        Returns:
            Batch of preprocessed tensors [B, C, H, W]
        """
        tensors = [self.preprocess(img) for img in image_inputs]
        batch = torch.stack(tensors)
        
        return batch
    
    def extract_features(
        self,
        image_input: Union[str, bytes, np.ndarray, Image.Image],
        model: torch.nn.Module
    ) -> np.ndarray:
        """
        Extract features from an image using a pre-trained model.
        
        Args:
            image_input: Image input
            model: PyTorch model for feature extraction
            
        Returns:
            Feature array
        """
        # Preprocess image
        tensor = self.preprocess(image_input)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = model(tensor)
        
        # Convert to numpy
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        return features