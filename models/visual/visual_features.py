# models/visual/visual_features.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image
import numpy as np
from enum import Enum
from transformers import CLIPFeatureExtractor, CLIPVisionModel

class VisualFeatureTypes(str, Enum):
    """Enumeration of supported visual feature types."""
    POOLED = "pooled"
    PATCH = "patch"
    CLS = "cls"
    ATTENTION = "attention"
    REGION = "region"

class VisualFeatureExtractor:
    """
    Extract visual features from images using pre-trained vision models.
    Supports various extraction methods and feature types.
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        feature_type: Union[VisualFeatureTypes, str] = VisualFeatureTypes.POOLED,
        normalize: bool = True
    ):
        """
        Initialize the visual feature extractor.
        
        Args:
            model_name: Name of the pre-trained model
            device: Device to run the model on
            feature_type: Type of features to extract
            normalize: Whether to normalize features
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set feature extraction parameters
        if isinstance(feature_type, VisualFeatureTypes):
            self.feature_type = feature_type.value
        else:
            self.feature_type = feature_type
            
        self.normalize = normalize
        
        # Load feature extractor and model
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Track feature dimensions
        if feature_type == "pooled":
            self.feature_dim = self.model.config.hidden_size
        elif feature_type == "patch":
            # Depends on image size and patch size
            # Will be determined dynamically
            self.feature_dim = self.model.config.hidden_size
        elif feature_type == "cls":
            self.feature_dim = self.model.config.hidden_size
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    
    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess images for the model.
        
        Args:
            images: Single image or list of images
            
        Returns:
            Dictionary of preprocessed inputs
        """
        # Convert single image to list
        if not isinstance(images, list):
            images = [images]
        
        # Preprocess images
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def extract_features(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            images: Single image or list of images
            
        Returns:
            Feature tensor [batch_size, feature_dim] or 
            [batch_size, num_patches, feature_dim] for patch features
        """
        # Preprocess images
        inputs = self.preprocess(images)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get features based on feature type
        if self.feature_type == "pooled":
            features = outputs.pooler_output
        elif self.feature_type == "patch":
            # Get patch features from last hidden state
            features = outputs.last_hidden_state
        elif self.feature_type == "cls":
            # Get CLS token features from last hidden state
            features = outputs.last_hidden_state[:, 0, :]
        
        # Normalize features if requested
        if self.normalize:
            if self.feature_type == "patch":
                # Normalize each patch embedding separately
                features = F.normalize(features, p=2, dim=2)
            else:
                # Normalize pooled or cls features
                features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def batch_extract_features(
        self,
        images: List[Image.Image],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Extract features from a large batch of images.
        
        Args:
            images: List of images
            batch_size: Batch size for processing
            
        Returns:
            Feature tensor for all images
        """
        all_features = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_features = self.extract_features(batch_images)
            all_features.append(batch_features)
        
        # Concatenate all features
        return torch.cat(all_features, dim=0)
    
    def get_attention_maps(
        self,
        images: Union[Image.Image, List[Image.Image]],
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Get attention maps from the vision model.
        
        Args:
            images: Single image or list of images
            layer_idx: Index of the attention layer
            
        Returns:
            Attention maps [batch_size, num_heads, seq_len, seq_len]
        """
        # Preprocess images
        inputs = self.preprocess(images)
        
        # Get attention outputs
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get attention maps from the specified layer
        # Default to last layer
        attention_maps = outputs.attentions[layer_idx]
        
        return attention_maps
    
    def visualize_attention(
        self,
        image: Image.Image,
        layer_idx: int = -1,
        head_idx: int = 0
    ) -> np.ndarray:
        """
        Visualize attention maps overlaid on the image.
        
        Args:
            image: Input image
            layer_idx: Index of the attention layer
            head_idx: Index of the attention head
            
        Returns:
            Visualization array (RGB format)
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Get attention maps
        attention_maps = self.get_attention_maps(image, layer_idx)
        
        # Select head
        attention_map = attention_maps[0, head_idx].cpu().numpy()
        
        # Get attention for the cls token to all other tokens
        cls_attention = attention_map[0, 1:]  # Skip cls token itself
        
        # Reshape to square for visualization
        # Approximate based on feature map size
        map_size = int(np.sqrt(len(cls_attention)))
        attention_grid = cls_attention[:map_size**2].reshape(map_size, map_size)
        
        # Resize attention map to match image size
        from scipy.ndimage import zoom
        zoom_factor = max(image.size) / map_size
        attention_upsampled = zoom(attention_grid, zoom_factor, order=1)
        
        # Crop to match image size
        height, width = attention_upsampled.shape
        img_width, img_height = image.size
        if height > img_height:
            attention_upsampled = attention_upsampled[:img_height, :]
        if width > img_width:
            attention_upsampled = attention_upsampled[:, :img_width]
        
        # Normalize attention map
        attention_upsampled = (attention_upsampled - attention_upsampled.min()) / (attention_upsampled.max() - attention_upsampled.min())
        
        # Convert image to array
        img_array = np.array(image)
        
        # Create heatmap and overlay
        heatmap = cm.jet(attention_upsampled)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Resize heatmap to match image size
        if heatmap.shape[:2] != img_array.shape[:2]:
            from skimage.transform import resize
            heatmap = resize(heatmap, img_array.shape[:2] + (3,), preserve_range=True).astype(np.uint8)
        
        # Blend image and heatmap
        alpha = 0.5
        blended = (alpha * heatmap + (1 - alpha) * img_array).astype(np.uint8)
        
        return blended
    
    def extract_region_features(
        self,
        image: Image.Image,
        regions: List[Tuple[int, int, int, int]]  # List of (x1, y1, x2, y2) coordinates
    ) -> torch.Tensor:
        """
        Extract features from specific regions of the image.
        
        Args:
            image: Input image
            regions: List of region coordinates (x1, y1, x2, y2)
            
        Returns:
            Feature tensor for each region [num_regions, feature_dim]
        """
        region_features = []
        
        # Process each region
        for x1, y1, x2, y2 in regions:
            # Crop region from image
            region = image.crop((x1, y1, x2, y2))
            
            # Extract features
            features = self.extract_features(region)
            
            # Add to list
            region_features.append(features)
        
        # Concatenate features
        return torch.cat(region_features, dim=0)
    
    def classify_image(
        self,
        image: Image.Image,
        categories: List[str],
        clip_model_name: str = "openai/clip-vit-base-patch32"
    ) -> Dict[str, float]:
        """
        Classify image into categories using CLIP model.
        
        Args:
            image: Input image
            categories: List of category names
            clip_model_name: CLIP model to use
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        from transformers import CLIPProcessor, CLIPModel
        
        # Load CLIP model and processor
        processor = CLIPProcessor.from_pretrained(clip_model_name)
        clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        
        # Prepare inputs
        inputs = processor(
            text=categories,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Compute similarity
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
        
        # Create dictionary mapping categories to probabilities
        results = {categories[i]: probs[i].item() for i in range(len(categories))}
        
        return results