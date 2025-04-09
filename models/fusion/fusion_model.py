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
        self.num_engagement_classes = num_engagement_classes
        
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
        
        # Categorize sentiments
        sentiment_categories = []
        for score in sentiment_scores.cpu().numpy():
            if score < -0.33:
                sentiment_categories.append("negative")
            elif score > 0.33:
                sentiment_categories.append("positive")
            else:
                sentiment_categories.append("neutral")
        
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
        Extract content features for retrieval.
        
        Args:
            visual_features: Visual features [batch_size, visual_seq_len, visual_dim]
            text_features: Text features [batch_size, text_seq_len, text_dim]
            visual_padding_mask: Visual padding mask [batch_size, visual_seq_len]
            text_padding_mask: Text padding mask [batch_size, text_seq_len]
            
        Returns:
            Content feature embeddings as numpy array
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
        
    def fine_tune(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        patience: int = 3,
        scheduler_type: str = 'cosine',
        warmup_steps: int = 100,
        gradient_clipping: float = 1.0,
        checkpoint_dir: Optional[str] = None,
        log_every: int = 10
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the fusion model on domain-specific data.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            patience: Patience for early stopping
            scheduler_type: Type of learning rate scheduler ('cosine', 'linear', 'step', None)
            warmup_steps: Number of warmup steps for schedulers that support it
            gradient_clipping: Maximum gradient norm for clipping
            checkpoint_dir: Directory to save checkpoints (None to disable)
            log_every: Log progress every N batches
            
        Returns:
            Dictionary containing training history (losses and metrics)
        """
        import torch.optim as optim
        from tqdm import tqdm
        import os
        import numpy as np
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Set model to training mode
        self.train()
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize loss function based on task type
        if self.engagement_type == "regression":
            criterion = torch.nn.MSELoss()
        else:  # classification
            criterion = torch.nn.CrossEntropyLoss()
        
        # Initialize learning rate scheduler
        scheduler = None
        if scheduler_type == 'cosine':
            from transformers import get_cosine_schedule_with_warmup
            num_training_steps = len(train_loader) * epochs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'linear':
            from transformers import get_linear_schedule_with_warmup
            num_training_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=5, 
                gamma=0.5
            )
        
        # Create checkpoint directory if needed
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize variables for tracking progress
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Initialize history dictionary
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'val_r2': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            running_loss = 0.0
            
            # Progress bar for training
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for i, batch in enumerate(train_iterator):
                # Get data
                visual_features = batch['visual_features'].to(self.device)
                text_features = batch['text_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(visual_features, text_features)
                
                # Get predicted values
                if self.engagement_type == "regression":
                    predictions = outputs["engagement"]["score"].squeeze(-1)
                else:  # classification
                    predictions = outputs["engagement"]["logits"]
                
                # Calculate loss
                loss = criterion(predictions, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clipping)
                
                # Update weights
                optimizer.step()
                
                # Update learning rate for batch-based schedulers
                if scheduler and scheduler_type in ['cosine', 'linear']:
                    scheduler.step()
                
                # Update running loss
                running_loss += loss.item() * visual_features.size(0)
                
                # Update progress bar
                train_iterator.set_postfix(loss=loss.item())
                
                # Log progress
                if i % log_every == 0:
                    print(f"Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Update epoch-based scheduler
            if scheduler and scheduler_type == 'step':
                scheduler.step()
            
            # Calculate epoch loss
            epoch_loss = running_loss / len(train_loader.dataset)
            history['train_loss'].append(epoch_loss)
            
            # Validation phase
            val_loss, val_mse, val_r2, _, _ = self._validate(val_loader, criterion)
            
            # Record validation metrics
            history['val_loss'].append(val_loss)
            history['val_mse'].append(val_mse)
            history['val_r2'].append(val_r2)
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {epoch_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model checkpoint
                if checkpoint_dir:
                    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                    self.save(best_model_path)
                    print(f"  Saved best model to {best_model_path}")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} epochs")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            
            # Save periodic checkpoint
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_loss,
                    'val_loss': val_loss
                }, checkpoint_path)
        
        # Return training history
        return history

    def _validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Validate the model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            Tuple of (val_loss, mse, r2, predictions, labels)
        """
        from tqdm import tqdm
        import numpy as np
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Set model to evaluation mode
        self.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Disable gradient computation for validation
        with torch.no_grad():
            # Progress bar for validation
            val_iterator = tqdm(val_loader, desc="Validation")
            
            for batch in val_iterator:
                # Get data
                visual_features = batch['visual_features'].to(self.device)
                text_features = batch['text_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self(visual_features, text_features)
                
                # Get predicted values
                if self.engagement_type == "regression":
                    predictions = outputs["engagement"]["score"].squeeze(-1)
                else:  # classification
                    predictions = torch.argmax(outputs["engagement"]["probabilities"], dim=1).float()
                
                # Calculate loss
                loss = criterion(predictions, labels)
                
                # Update running loss
                running_loss += loss.item() * visual_features.size(0)
                
                # Collect predictions and labels for metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation loss
        val_loss = running_loss / len(val_loader.dataset)
        
        # Calculate regression metrics
        mse = mean_squared_error(all_labels, all_predictions)
        r2 = r2_score(all_labels, all_predictions)
        
        return val_loss, mse, r2, np.array(all_predictions), np.array(all_labels)