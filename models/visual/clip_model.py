# models/visual/clip_model.py
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

class CLIPWrapper(nn.Module):
    """
    Wrapper for the CLIP model for visual content analysis.
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None
    ):
        """
        Initialize the CLIP wrapper.
        
        Args:
            model_name: Name of the CLIP model to use
            device: Device to run the model on (cpu, cuda, auto)
        """
        super().__init__()
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Move model to device
        self.model.to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
    
    def encode_images(
        self,
        images: Union[Image.Image, List[Image.Image]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode images into embeddings.
        
        Args:
            images: Single image or list of images
            normalize: Whether to normalize embeddings
            
        Returns:
            Image embeddings
        """
        # Handle single image
        if not isinstance(images, list):
            images = [images]
        
        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values)
        
        # Normalize if requested
        if normalize:
            image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text into embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings
            
        Returns:
            Text embeddings
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Process text
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Get text features
        with torch.no_grad():
            text_features = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Normalize if requested
        if normalize:
            text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features
    
    def compute_similarity(
        self,
        images: Union[Image.Image, List[Image.Image]],
        texts: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Compute similarity between images and texts.
        
        Args:
            images: Single image or list of images
            texts: Single text or list of texts
            
        Returns:
            Similarity matrix
        """
        # Encode images and text
        image_embeddings = self.encode_images(images)
        text_embeddings = self.encode_text(texts)
        
        # Compute similarity
        similarity = torch.matmul(image_embeddings, text_embeddings.t())
        
        return similarity
    
    def classify_images(
        self,
        images: Union[Image.Image, List[Image.Image]],
        categories: List[str]
    ) -> Dict[str, Any]:
        """
        Classify images into categories.
        
        Args:
            images: Single image or list of images
            categories: List of category names
            
        Returns:
            Dictionary with classification results
        """
        # Compute similarity
        similarity = self.compute_similarity(images, categories)
        
        # Get probabilities
        probs = F.softmax(similarity, dim=1)
        
        # Get top categories
        values, indices = torch.topk(probs, k=min(3, len(categories)), dim=1)
        
        # Format results
        results = []
        batch_size = probs.shape[0]
        
        for i in range(batch_size):
            top_cats = [
                {
                    "category": categories[indices[i, j].item()],
                    "confidence": values[i, j].item()
                }
                for j in range(indices.shape[1])
            ]
            
            results.append({
                "top_categories": top_cats,
                "all_scores": {cat: probs[i, j].item() for j, cat in enumerate(categories)}
            })
        
        # Return single result for single image
        if len(results) == 1 and not isinstance(images, list):
            return results[0]
        
        return {"batch_results": results}
    
    def extract_visual_attributes(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> Dict[str, Any]:
        """
        Extract visual attributes from images.
        
        Args:
            images: Single image or list of images
            
        Returns:
            Dictionary with attribute analysis
        """
        # Predefined visual attributes
        attributes = {
            "style": ["realistic", "cartoon", "abstract", "graphic", "artistic"],
            "color": ["vibrant", "muted", "dark", "bright", "monochrome"],
            "composition": ["close-up", "wide-shot", "detailed", "minimal", "centered"],
            "emotion": ["happy", "sad", "exciting", "calm", "tense", "neutral"],
            "quality": ["professional", "amateur", "high-quality", "low-quality"]
        }
        
        # Flatten attributes for classification
        all_attrs = []
        attr_mapping = {}
        
        for category, attrs in attributes.items():
            for attr in attrs:
                attr_full = f"{category}: {attr}"
                all_attrs.append(attr_full)
                attr_mapping[attr_full] = (category, attr)
        
        # Classify images
        results = self.classify_images(images, all_attrs)
        
        # Reformat results by attribute category
        if "batch_results" in results:
            # Multiple images
            formatted_results = []
            
            for single_result in results["batch_results"]:
                formatted = self._format_attribute_result(single_result, attr_mapping, attributes)
                formatted_results.append(formatted)
            
            return {"batch_results": formatted_results}
        else:
            # Single image
            return self._format_attribute_result(results, attr_mapping, attributes)
    
    def _format_attribute_result(
        self,
        result: Dict[str, Any],
        attr_mapping: Dict[str, Tuple[str, str]],
        attributes: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Format attribute results by category."""
        formatted = {"attributes": {}}
        
        # Group by attribute category
        for category in attributes.keys():
            category_scores = {}
            
            for attr_full, score in result["all_scores"].items():
                cat, attr = attr_mapping[attr_full]
                if cat == category:
                    category_scores[attr] = score
            
            # Get top attribute for this category
            top_attr = max(category_scores.items(), key=lambda x: x[1])
            
            formatted["attributes"][category] = {
                "top": top_attr[0],
                "confidence": top_attr[1],
                "all_scores": category_scores
            }
        
        # Add overall top attributes
        formatted["top_attributes"] = [
            {
                "category": attr_mapping[cat["category"]][0],
                "attribute": attr_mapping[cat["category"]][1],
                "confidence": cat["confidence"]
            }
            for cat in result["top_categories"]
        ]
        
        return formatted