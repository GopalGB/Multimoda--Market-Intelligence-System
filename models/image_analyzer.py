# models/image_analyzer.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Dict, Union, Any
import os

class ProductImageAnalyzer:
    """Analyzes product images using CLIP model from Hugging Face."""
    
    def __init__(self):
        """Initialize with CLIP model."""
        # Load model
        print("Loading CLIP model... (this may take a moment)")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Default categories for classification
        self.default_categories = [
            "luxury premium product",
            "budget affordable product",
            "eco-friendly sustainable product",
            "traditional conventional product",
            "innovative cutting-edge product",
            "family-oriented product",
            "minimalist simple product",
            "feature-rich complex product"
        ]
        
        # Default product attributes
        self.default_attributes = [
            "high quality",
            "low quality",
            "modern design",
            "traditional design",
            "bright colorful",
            "neutral subdued",
            "technical advanced",
            "simple basic"
        ]
    
    def analyze_image(
        self, 
        image_path: str, 
        categories: List[str] = None,
        attributes: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a product image against categories and attributes.
        """
        # Load image
        if isinstance(image_path, str) and os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            # Handle case where image_path might be a file object
            try:
                image = Image.open(image_path)
            except:
                raise ValueError("Invalid image path or file")
        
        # Use default categories if none provided
        if categories is None:
            categories = self.default_categories
            
        # Use default attributes if none provided
        if attributes is None:
            attributes = self.default_attributes
        
        # Analyze categories
        category_results = self._classify_image(image, categories)
        
        # Analyze attributes
        attribute_results = self._classify_image(image, attributes)
        
        # Combine results
        results = {
            "category_analysis": category_results,
            "attribute_analysis": attribute_results,
            "top_category": max(category_results.items(), key=lambda x: x[1])[0],
            "top_attributes": [k for k, v in sorted(attribute_results.items(), key=lambda x: x[1], reverse=True)[:3]]
        }
        
        return results
    
    def _classify_image(self, image: Image.Image, text_labels: List[str]) -> Dict[str, float]:
        """
        Classify image against text labels using CLIP.
        """
        # Prepare inputs
        inputs = self.processor(
            text=text_labels, 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Process results
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze(0)
        
        # Create results dictionary
        results = {text_labels[i]: float(probs[i]) for i in range(len(text_labels))}
        
        return results