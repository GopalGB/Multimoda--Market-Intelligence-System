"""
Model Manager for Audience Intelligence API.
Handles loading, caching, and running models.
"""
import os
import logging
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Path setup
TEXT_MODEL_PATH = os.getenv('TEXT_MODEL_PATH', 'models/saved/roberta_model.pt')
VISUAL_MODEL_PATH = os.getenv('VISUAL_MODEL_PATH', 'models/saved/clip_model.pt')
FUSION_MODEL_PATH = os.getenv('FUSION_MODEL_PATH', 'models/saved/fusion_model_best.pt')
CAUSAL_MODEL_PATH = os.getenv('CAUSAL_MODEL_PATH', 'models/saved/causal_model.pt')

class ModelManager:
    """Manager for all models used in the Audience Intelligence API."""
    
    def __init__(self):
        """Initialize the model manager with lazy loading of models."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model holders
        self.text_model = None
        self.visual_model = None
        self.fusion_model = None
        self.causal_model = None
        
        # Cache for analysis results
        self.analysis_cache = {}
        
        # Keep track of loaded models
        self.loaded_models = []
        
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        return self.loaded_models
    
    def load_text_model(self):
        """Load the text embedding model."""
        if self.text_model is not None:
            return
        
        try:
            logger.info(f"Loading text model from {TEXT_MODEL_PATH}")
            # Import only when needed
            from models.text.roberta_model import RoBERTaWrapper
            
            # Initialize model
            self.text_model = RoBERTaWrapper(device=self.device)
            
            # Add to loaded models list
            self.loaded_models.append('text_model')
            logger.info("Text model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading text model: {e}")
            raise RuntimeError(f"Failed to load text model: {e}")
    
    def load_visual_model(self):
        """Load the visual embedding model."""
        if self.visual_model is not None:
            return
        
        try:
            logger.info(f"Loading visual model from {VISUAL_MODEL_PATH}")
            # Import only when needed
            from models.visual.clip_model import CLIPWrapper
            
            # Initialize model
            self.visual_model = CLIPWrapper(device=self.device)
            
            # Add to loaded models list
            self.loaded_models.append('visual_model')
            logger.info("Visual model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading visual model: {e}")
            raise RuntimeError(f"Failed to load visual model: {e}")
    
    def load_fusion_model(self):
        """Load the multimodal fusion model."""
        if self.fusion_model is not None:
            return
        
        try:
            logger.info(f"Loading fusion model from {FUSION_MODEL_PATH}")
            # Import only when needed
            from models.fusion.fusion_model import MultimodalFusionModel
            
            # Check if file exists
            if not os.path.exists(FUSION_MODEL_PATH):
                raise FileNotFoundError(f"Fusion model file not found: {FUSION_MODEL_PATH}")
            
            # Load model
            self.fusion_model = MultimodalFusionModel.load(FUSION_MODEL_PATH, device=self.device)
            
            # Add to loaded models list
            self.loaded_models.append('fusion_model')
            logger.info("Fusion model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading fusion model: {e}")
            raise RuntimeError(f"Failed to load fusion model: {e}")
    
    def load_causal_model(self):
        """Load the causal model."""
        if self.causal_model is not None:
            return
        
        try:
            logger.info(f"Loading causal model from {CAUSAL_MODEL_PATH}")
            # Import only when needed
            from causal.structural_model import StructuralCausalModel
            
            # Initialize model
            self.causal_model = StructuralCausalModel()
            
            # Load model if file exists
            if os.path.exists(CAUSAL_MODEL_PATH):
                self.causal_model.load(CAUSAL_MODEL_PATH)
            
            # Add to loaded models list
            self.loaded_models.append('causal_model')
            logger.info("Causal model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading causal model: {e}")
            raise RuntimeError(f"Failed to load causal model: {e}")
    
    def extract_text_features(self, text: str) -> torch.Tensor:
        """Extract text features from input text."""
        # Load text model if not loaded
        if self.text_model is None:
            self.load_text_model()
        
        # Process text and return features
        return self.text_model.encode_text(text)
    
    def extract_visual_features(self, image: Image.Image) -> torch.Tensor:
        """Extract visual features from input image."""
        # Load visual model if not loaded
        if self.visual_model is None:
            self.load_visual_model()
        
        # Process image and return features
        return self.visual_model.encode_images(image)
    
    def run_fusion_model(
        self, 
        text_features: torch.Tensor, 
        visual_features: torch.Tensor
    ) -> Dict[str, Any]:
        """Run the fusion model to get content analysis."""
        # Load fusion model if not loaded
        if self.fusion_model is None:
            self.load_fusion_model()
        
        # Run model
        outputs = self.fusion_model(visual_features, text_features)
        
        # Process outputs
        if self.fusion_model.engagement_type == "regression":
            engagement_score = outputs["engagement"]["score"].squeeze(-1).item()
        else:
            # For classification, convert class probabilities to a score
            probs = outputs["engagement"]["probabilities"]
            class_scores = torch.arange(0, 1.01, 1.0/probs.size(1), device=probs.device)
            engagement_score = torch.sum(probs * class_scores).item()
        
        # Get sentiment
        sentiment_score = outputs["sentiment"].squeeze(-1).item()
        
        # Determine sentiment category
        if sentiment_score > 0.33:
            sentiment_category = "positive"
        elif sentiment_score < -0.33:
            sentiment_category = "negative"
        else:
            sentiment_category = "neutral"
        
        # Extract content features for causal analysis
        content_features = outputs["content_features"].detach().cpu()
        
        # Return processed results
        return {
            "engagement_score": engagement_score,
            "sentiment_score": sentiment_score,
            "sentiment_category": sentiment_category,
            "content_features": content_features
        }
    
    def run_text_only_prediction(self, text_features: torch.Tensor) -> Dict[str, Any]:
        """Run a text-only prediction as fallback."""
        # This is a simplified model for when images are not available
        
        # Create a normalized feature 
        normalized = torch.nn.functional.normalize(text_features, p=2, dim=1)
        
        # Simplified engagement score estimation using text features
        # This would normally use a trained regression model
        feature_mean = torch.mean(normalized).item()
        engagement_score = max(0, min(1, 0.5 + feature_mean))
        
        # Simplified sentiment analysis
        # In a real implementation, this would use a sentiment classifier
        sentiment_score = feature_mean * 2  # Scale to roughly -1 to 1
        
        # Determine sentiment category
        if sentiment_score > 0.33:
            sentiment_category = "positive"
        elif sentiment_score < -0.33:
            sentiment_category = "negative"
        else:
            sentiment_category = "neutral"
        
        # Return processed results
        return {
            "engagement_score": engagement_score,
            "sentiment_score": sentiment_score,
            "sentiment_category": sentiment_category,
            "content_features": text_features.detach().cpu()
        }
    
    def extract_causal_factors(
        self,
        text: str,
        content_features: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """Extract causal factors that influence engagement."""
        # Load causal model if not loaded
        if self.causal_model is None:
            self.load_causal_model()
        
        # For demonstration, we'll return some fixed causal factors
        # In a real implementation, this would use the causal model
        
        # Extract content characteristics from text
        word_count = len(text.split())
        avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
        
        # Sample causal factors (would be derived from model in production)
        factors = [
            {
                "name": "Content Length",
                "value": min(1.0, word_count / 1000),  # Normalize to 0-1
                "effect": 0.35 if word_count > 500 else 0.15,
                "direction": "positive" if word_count > 300 else "negative"
            },
            {
                "name": "Emotional Words",
                "value": 0.65,  # Would be calculated from text
                "effect": 0.28,
                "direction": "positive"
            },
            {
                "name": "Visual Appeal",
                "value": 0.72 if content_features is not None else 0.0,  # Based on image presence
                "effect": 0.42 if content_features is not None else 0.0,
                "direction": "positive"
            },
            {
                "name": "Clarity",
                "value": 0.8 if avg_word_length < 6 else 0.4,
                "effect": 0.22,
                "direction": "positive" if avg_word_length < 6 else "negative"
            },
            {
                "name": "Topic Relevance",
                "value": 0.65,  # Would be calculated
                "effect": 0.31,
                "direction": "positive"
            }
        ]
        
        # Sort by effect magnitude
        factors.sort(key=lambda x: abs(x["effect"]), reverse=True)
        
        return factors
    
    def generate_causal_graph(
        self,
        text: str,
        text_features: torch.Tensor
    ) -> Dict[str, Any]:
        """Generate a causal graph for visualization."""
        # This would use the causal model to create a graph structure
        
        # For demonstration, return a simple causal graph
        nodes = [
            {"id": "content_length", "name": "Content Length", "type": "cause"},
            {"id": "emotional_words", "name": "Emotional Words", "type": "cause"},
            {"id": "visual_appeal", "name": "Visual Appeal", "type": "cause"},
            {"id": "clarity", "name": "Clarity", "type": "cause"},
            {"id": "topic_relevance", "name": "Topic Relevance", "type": "cause"},
            {"id": "engagement", "name": "Engagement", "type": "effect"}
        ]
        
        edges = [
            {"source": "content_length", "target": "engagement", "weight": 0.35},
            {"source": "emotional_words", "target": "engagement", "weight": 0.28},
            {"source": "visual_appeal", "target": "engagement", "weight": 0.42},
            {"source": "clarity", "target": "engagement", "weight": 0.22},
            {"source": "topic_relevance", "target": "engagement", "weight": 0.31},
            {"source": "emotional_words", "target": "clarity", "weight": -0.15},
            {"source": "content_length", "target": "clarity", "weight": -0.2}
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "layout": "dagre"  # Suggest layout algorithm for frontend
        }
    
    def generate_suggestions(
        self, 
        causal_factors: List[Dict[str, Any]],
        text: str,
        title: str
    ) -> List[str]:
        """Generate content optimization suggestions."""
        suggestions = []
        
        # Find factors with negative or weak effects
        for factor in causal_factors:
            if factor["direction"] == "negative":
                if factor["name"] == "Content Length":
                    if len(text.split()) > 1000:
                        suggestions.append("Consider shortening your content to improve readability.")
                    else:
                        suggestions.append("Consider adding more detailed information to increase engagement.")
                elif factor["name"] == "Clarity":
                    suggestions.append("Use simpler language and shorter sentences to improve clarity.")
                elif factor["name"] == "Visual Appeal" and factor["value"] < 0.3:
                    suggestions.append("Add relevant images to increase visual appeal.")
            elif factor["direction"] == "positive" and factor["effect"] > 0.3 and factor["value"] < 0.5:
                if factor["name"] == "Emotional Words":
                    suggestions.append("Use more emotional and engaging language.")
                elif factor["name"] == "Topic Relevance":
                    suggestions.append("Focus more clearly on the main topic in your content.")
        
        # Add title suggestion if title is short
        if len(title.split()) < 4:
            suggestions.append("Use a more descriptive title to increase reader interest.")
        
        # Add a suggestion about structure
        if len(text) > 1000 and "<h" not in text.lower() and "\n\n" not in text:
            suggestions.append("Break up content with subheadings or paragraphs for better readability.")
        
        return suggestions
    
    def generate_optimization_suggestions(
        self,
        text: str,
        title: str,
        text_features: torch.Tensor
    ) -> Dict[str, List[str]]:
        """Generate detailed optimization suggestions."""
        # This would integrate with a more sophisticated suggestion engine
        
        # Extract some basic content metrics
        word_count = len(text.split())
        avg_sentence_length = len(text) / max(1, text.count('.') + text.count('!') + text.count('?'))
        
        # Generate suggestions
        title_suggestions = []
        content_suggestions = []
        
        # Title suggestions
        if len(title.split()) < 4:
            title_suggestions.append("Use a more descriptive and compelling title")
        if not any(word.lower() in title.lower() for word in ["how", "why", "what", "when", "top", "best"]):
            title_suggestions.append("Consider using question words or superlatives in your title")
        
        # Content suggestions
        if word_count < 300:
            content_suggestions.append("Add more detailed content to increase engagement")
        elif word_count > 1500:
            content_suggestions.append("Consider breaking very long content into separate articles")
        
        if avg_sentence_length > 25:
            content_suggestions.append("Use shorter sentences to improve readability")
        
        if text.count('\n\n') < 3 and word_count > 500:
            content_suggestions.append("Break up content with more paragraph breaks")
        
        if text.lower().count('img') == 0 and text.lower().count('image') == 0:
            content_suggestions.append("Add relevant images to increase visual appeal")
        
        # Return organized suggestions
        return {
            "title": title_suggestions,
            "content": content_suggestions,
            "estimated_improvement": 0.15 if len(title_suggestions) + len(content_suggestions) > 2 else 0.05
        }
    
    def generate_counterfactual(
        self,
        content_id: str,
        factor_name: str,
        factor_value: float
    ) -> Dict[str, Any]:
        """Generate a counterfactual prediction."""
        # Check if content exists in cache
        if content_id not in self.analysis_cache:
            raise ValueError(f"Content ID not found: {content_id}")
        
        # Get original analysis
        original = self.analysis_cache[content_id]
        original_score = original.get('engagement_score', 0.5)
        
        # Find factor in causal factors
        causal_factors = self.extract_causal_factors(
            text=original.get('text', ''),
            content_features=original.get('content_features', None)
        )
        
        factor = None
        for f in causal_factors:
            if f["name"].lower() == factor_name.lower():
                factor = f
                break
        
        if factor is None:
            raise ValueError(f"Factor not found: {factor_name}")
        
        # Calculate counterfactual effect
        original_value = factor.get('value', 0.5)
        effect_magnitude = factor.get('effect', 0.1)
        direction = 1 if factor.get('direction', 'positive') == 'positive' else -1
        
        # Adjust score based on the change in factor value
        value_change = factor_value - original_value
        score_change = value_change * effect_magnitude * direction
        counterfactual_score = max(0, min(1, original_score + score_change))
        
        # Return results
        return {
            'original_score': original_score,
            'counterfactual_score': counterfactual_score,
            'difference': counterfactual_score - original_score,
            'original_value': original_value,
            'factor': factor
        }
    
    def store_analysis_result(
        self,
        url: str,
        title: str,
        features: Dict[str, Any]
    ) -> str:
        """Store analysis result in cache for later reference."""
        # Generate a unique ID
        content_id = str(uuid.uuid4())
        
        # Store in cache
        self.analysis_cache[content_id] = {
            'url': url,
            'title': title,
            'engagement_score': features.get('engagement_score', 0),
            'sentiment_score': features.get('sentiment_score', 0),
            'content_features': features.get('content_features', None),
            'timestamp': str(datetime.now().isoformat())
        }
        
        # Clean cache if too large (keep only 100 most recent)
        if len(self.analysis_cache) > 100:
            oldest_id = min(self.analysis_cache.keys(), key=lambda k: self.analysis_cache[k]['timestamp'])
            del self.analysis_cache[oldest_id]
        
        return content_id 