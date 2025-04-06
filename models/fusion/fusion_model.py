# models/fusion/fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from .cross_attention import CrossModalTransformer

class MultimodalFusionModel(nn.Module):
    """
    Complete multimodal fusion model for audience intelligence.
    Combines cross-modal transformer with task-specific prediction heads.
    """
    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 768,
        fusion_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        num_engagement_classes: int = 5,  # For engagement level prediction
        engagement_type: str = "classification",  # "classification" or "regression"
        device: Optional[str] = None
    ):
        """
        Initialize the multimodal fusion model.
        
        Args:
            visual_dim: Dimension of visual features
            text_dim: Dimension of text features
            fusion_dim: Dimension for attention mechanisms
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            feedforward_dim: Dimension of feedforward network
            dropout: Dropout probability
            num_engagement_classes: Number of engagement level classes
            engagement_type: Type of engagement prediction task
            device: Device to run the model on
        """
        super().__init__()
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.engagement_type = engagement_type
        
        # Cross-modal transformer for fusion
        self.fusion_transformer = CrossModalTransformer(
            visual_dim=visual_dim,
            text_dim=text_dim,
            fusion_dim=fusion_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            fusion_output_dim=fusion_dim
        )
        
        # Task-specific heads
        
        # 1. Engagement prediction head
        if engagement_type == "classification":
            self.engagement_head = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, num_engagement_classes)
            )
        else:  # regression
            self.engagement_head = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, 1)
            )
            
        # 2. Sentiment prediction head
        self.sentiment_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1)  # Sentiment score from -1 to 1
        )
        
        # 3. Content feature prediction head
        self.content_feature_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass for the multimodal fusion model.
        
        Args:
            visual_features: Visual features [batch_size, visual_seq_len, visual_dim]
            text_features: Text features [batch_size, text_seq_len, text_dim]
            visual_padding_mask: Visual padding mask [batch_size, visual_seq_len]
            text_padding_mask: Text padding mask [batch_size, text_seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with model outputs
        """
        # Apply fusion transformer
        fusion_outputs = self.fusion_transformer(
            visual_features,
            text_features,
            visual_padding_mask,
            text_padding_mask,
            return_attention=return_attention,
            return_individual_features=True
        )
        
        # Get fused features
        fused_features = fusion_outputs["fused_features"]
        
        # Apply task-specific heads
        
        # 1. Engagement prediction
        if self.engagement_type == "classification":
            engagement_logits = self.engagement_head(fused_features)
            engagement_probs = F.softmax(engagement_logits, dim=1)
            engagement_output = {
                "logits": engagement_logits,
                "probabilities": engagement_probs
            }
        else:  # regression
            engagement_pred = self.engagement_head(fused_features)
            engagement_output = {
                "score": engagement_pred
            }
            
        # 2. Sentiment prediction
        sentiment_pred = self.sentiment_head(fused_features)
        # Apply tanh to get values between -1 and 1
        sentiment_score = torch.tanh(sentiment_pred)
        
        # 3. Content feature extraction
        content_features = self.content_feature_head(fused_features)
        
        # Combine all outputs
        outputs = {
            "fused_features": fused_features,
            "engagement": engagement_output,
            "sentiment": sentiment_score,
            "content_features": content_features,
            "visual_representation": fusion_outputs["visual_representation"],
            "text_representation": fusion_outputs["text_representation"]
        }
        
        # Add attention maps if requested
        if return_attention and "attention_maps" in fusion_outputs:
            outputs["attention_maps"] = fusion_outputs["attention_maps"]
            
        return outputs
    
    def predict_engagement(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Predict engagement level for content.
        
        Args:
            visual_features: Visual features [batch_size, visual_seq_len, visual_dim]
            text_features: Text features [batch_size, text_seq_len, text_dim]
            visual_padding_mask: Visual padding mask [batch_size, visual_seq_len]
            text_padding_mask: Text padding mask [batch_size, text_seq_len]
            
        Returns:
            Dictionary with engagement predictions
        """
        # Forward pass
        outputs = self.forward(
            visual_features,
            text_features,
            visual_padding_mask,
            text_padding_mask,
            return_attention=False
        )
        
        # Extract engagement predictions
        engagement_output = outputs["engagement"]
        
        # Format based on task type
        if self.engagement_type == "classification":
            probs = engagement_output["probabilities"]
            classes = torch.argmax(probs, dim=1)
            
            # Map class indices to meaningful labels
            class_names = ["very_low", "low", "medium", "high", "very_high"]
            labels = [class_names[idx] for idx in classes.cpu().numpy()]
            
            return {
                "class_index": classes.cpu().numpy(),
                "class_label": labels,
                "probabilities": probs.cpu().numpy(),
                "confidence": torch.max(probs, dim=1)[0].cpu().numpy()
            }
        else:  # regression
            scores = engagement_output["score"].squeeze(-1)
            
            # Normalize to 0-1 range
            normalized_scores = torch.sigmoid(scores)
            
            return {
                "engagement_score": normalized_scores.cpu().numpy(),
                "raw_score": scores.cpu().numpy()
            }
    
    def predict_sentiment(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Predict sentiment for content.
        
        Args:
            visual_features: Visual features [batch_size, visual_seq_len, visual_dim]
            text_features: Text features [batch_size, text_seq_len, text_dim]
            visual_padding_mask: Visual padding mask [batch_size, visual_seq_len]
            text_padding_mask: Text padding mask [batch_size, text_seq_len]
            
        Returns:
            Dictionary with sentiment predictions
        """
        # Forward pass
        outputs = self.forward(
            visual_features,
            text_features,
            visual_padding_mask,
            text_padding_mask,
            return_attention=False
        )
        
        # Extract sentiment predictions
        sentiment_scores = outputs["sentiment"].squeeze(-1)
        
        # Map scores to sentiment categories
        sentiment_categories = []
        for score in sentiment_scores:
            score_val = score.item()
            if score_val < -0.6:
                sentiment_categories.append("very_negative")
            elif score_val < -0.2:
                sentiment_categories.append("negative")
            elif score_val < 0.2:
                sentiment_categories.append("neutral")
            elif score_val < 0.6:
                sentiment_categories.append("positive")
            else:
                sentiment_categories.append("very_positive")
        
        return {
            "sentiment_score": sentiment_scores.cpu().numpy(),
            "sentiment_category": sentiment_categories
        }
    
    def extract_content_features(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Extract content features for similarity comparison or clustering.
        
        Args:
            visual_features: Visual features [batch_size, visual_seq_len, visual_dim]
            text_features: Text features [batch_size, text_seq_len, text_dim]
            visual_padding_mask: Visual padding mask [batch_size, visual_seq_len]
            text_padding_mask: Text padding mask [batch_size, text_seq_len]
            
        Returns:
            Content feature embeddings
        """
        # Forward pass
        outputs = self.forward(
            visual_features,
            text_features,
            visual_padding_mask,
            text_padding_mask,
            return_attention=False
        )
        
        # Extract content features
        content_features = outputs["content_features"]
        
        # Normalize features for similarity comparison
        content_features = F.normalize(content_features, p=2, dim=1)
        
        return content_features.cpu().numpy()
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "visual_dim": self.fusion_transformer.layers[0].visual_self_attn.embed_dim,
                "text_dim": self.fusion_transformer.layers[0].text_self_attn.embed_dim,
                "fusion_dim": self.fusion_transformer.fusion_output[0].in_features // 2,
                "num_layers": len(self.fusion_transformer.layers),
                "num_heads": self.fusion_transformer.layers[0].visual_self_attn.num_heads,
                "engagement_type": self.engagement_type
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "MultimodalFusionModel":
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with saved configuration
        model = cls(
            **checkpoint["config"],
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model