# models/fusion/cross_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class MultiheadCrossAttention(nn.Module):
    """
    Multihead cross-attention module that allows one modality to attend to another.
    """
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize the multihead cross-attention module.
        
        Args:
            query_dim: Dimension of query features
            key_dim: Dimension of key/value features
            embed_dim: Dimension of the attention embedding
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_dim, embed_dim)
        self.value_proj = nn.Linear(key_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, query_dim)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize the projection layers with Xavier uniform
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        # Initialize biases to zero
        nn.init.constant_(self.query_proj.bias, 0.)
        nn.init.constant_(self.key_proj.bias, 0.)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.constant_(self.output_proj.bias, 0.)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for cross-attention.
        
        Args:
            query: Query tensor [batch_size, query_seq_len, query_dim]
            key: Key tensor [batch_size, key_seq_len, key_dim]
            value: Value tensor [batch_size, key_seq_len, key_dim]
            attention_mask: Optional mask tensor [batch_size, query_seq_len, key_seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attention output [batch_size, query_seq_len, query_dim]
            attention_weights: Optional attention weights [batch_size, num_heads, query_seq_len, key_seq_len]
        """
        batch_size, query_seq_len, _ = query.shape
        _, key_seq_len, _ = key.shape
        
        # Linear projections and reshape for multi-head attention
        query = self.query_proj(query).view(batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale dot-product attention
        # [batch_size, num_heads, query_seq_len, head_dim] @ [batch_size, num_heads, head_dim, key_seq_len]
        # -> [batch_size, num_heads, query_seq_len, key_seq_len]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Add the mask to attention scores (we want to apply -inf to masked positions)
            attention_scores = attention_scores + attention_mask.unsqueeze(1)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        # [batch_size, num_heads, query_seq_len, key_seq_len] @ [batch_size, num_heads, key_seq_len, head_dim]
        # -> [batch_size, num_heads, query_seq_len, head_dim]
        context = torch.matmul(attention_weights, value)
        
        # Reshape back to original dimensions
        # [batch_size, num_heads, query_seq_len, head_dim] -> [batch_size, query_seq_len, embed_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, query_seq_len, self.embed_dim)
        
        # Final linear projection
        output = self.output_proj(context)
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None


class CrossModalTransformerLayer(nn.Module):
    """
    A transformer layer with cross-modal attention that enables bidirectional
    information flow between visual and textual modalities.
    """
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        fusion_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        """
        Initialize the cross-modal transformer layer.
        
        Args:
            visual_dim: Dimension of visual features
            text_dim: Dimension of text features
            fusion_dim: Dimension for attention mechanisms
            num_heads: Number of attention heads
            feedforward_dim: Dimension of feedforward network
            dropout: Dropout probability
            layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()
        
        # Self-attention for each modality
        self.visual_self_attn = nn.MultiheadAttention(visual_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_self_attn = nn.MultiheadAttention(text_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention between modalities
        self.visual_to_text_attn = MultiheadCrossAttention(text_dim, visual_dim, fusion_dim, num_heads, dropout)
        self.text_to_visual_attn = MultiheadCrossAttention(visual_dim, text_dim, fusion_dim, num_heads, dropout)
        
        # Feedforward networks
        self.visual_ff = nn.Sequential(
            nn.Linear(visual_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, visual_dim)
        )
        
        self.text_ff = nn.Sequential(
            nn.Linear(text_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, text_dim)
        )
        
        # Layer normalization
        self.visual_norm1 = nn.LayerNorm(visual_dim, eps=layer_norm_eps)
        self.visual_norm2 = nn.LayerNorm(visual_dim, eps=layer_norm_eps)
        self.visual_norm3 = nn.LayerNorm(visual_dim, eps=layer_norm_eps)
        
        self.text_norm1 = nn.LayerNorm(text_dim, eps=layer_norm_eps)
        self.text_norm2 = nn.LayerNorm(text_dim, eps=layer_norm_eps)
        self.text_norm3 = nn.LayerNorm(text_dim, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the cross-modal transformer layer.
        
        Args:
            visual_features: Visual features [batch_size, visual_seq_len, visual_dim]
            text_features: Text features [batch_size, text_seq_len, text_dim]
            visual_padding_mask: Visual padding mask [batch_size, visual_seq_len]
            text_padding_mask: Text padding mask [batch_size, text_seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing updated features and optional attention weights
        """
        # Step 1: Self-attention within each modality
        visual_attn_out, visual_attn_weights = self.visual_self_attn(
            self.visual_norm1(visual_features),
            self.visual_norm1(visual_features),
            self.visual_norm1(visual_features),
            key_padding_mask=visual_padding_mask,
            need_weights=return_attention
        )
        visual_features = visual_features + self.dropout(visual_attn_out)
        
        text_attn_out, text_attn_weights = self.text_self_attn(
            self.text_norm1(text_features),
            self.text_norm1(text_features),
            self.text_norm1(text_features),
            key_padding_mask=text_padding_mask,
            need_weights=return_attention
        )
        text_features = text_features + self.dropout(text_attn_out)
        
        # Step 2: Cross-modal attention
        # Text attends to visual features
        text_cross_out, text_cross_attn = self.visual_to_text_attn(
            self.text_norm2(text_features),
            self.visual_norm2(visual_features),
            self.visual_norm2(visual_features),
            return_attention=return_attention
        )
        text_features = text_features + self.dropout(text_cross_out)
        
        # Visual attends to text features
        visual_cross_out, visual_cross_attn = self.text_to_visual_attn(
            self.visual_norm2(visual_features),
            self.text_norm2(text_features),
            self.text_norm2(text_features),
            return_attention=return_attention
        )
        visual_features = visual_features + self.dropout(visual_cross_out)
        
        # Step 3: Feedforward networks
        visual_ff_out = self.visual_ff(self.visual_norm3(visual_features))
        visual_features = visual_features + self.dropout(visual_ff_out)
        
        text_ff_out = self.text_ff(self.text_norm3(text_features))
        text_features = text_features + self.dropout(text_ff_out)
        
        # Return results
        result = {
            "visual_features": visual_features,
            "text_features": text_features
        }
        
        if return_attention:
            result.update({
                "visual_self_attention": visual_attn_weights,
                "text_self_attention": text_attn_weights,
                "visual_to_text_attention": text_cross_attn,
                "text_to_visual_attention": visual_cross_attn
            })
            
        return result


class CrossModalTransformer(nn.Module):
    """
    Full cross-modal transformer with multiple layers for fusing visual and text features.
    """
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        fusion_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        fusion_output_dim: Optional[int] = None
    ):
        """
        Initialize the cross-modal transformer.
        
        Args:
            visual_dim: Dimension of visual features
            text_dim: Dimension of text features
            fusion_dim: Dimension for attention mechanisms
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            feedforward_dim: Dimension of feedforward network
            dropout: Dropout probability
            fusion_output_dim: Optional output dimension after fusion (if None, uses visual_dim + text_dim)
        """
        super().__init__()
        
        # Create multiple layers of cross-modal transformers
        self.layers = nn.ModuleList([
            CrossModalTransformerLayer(
                visual_dim=visual_dim,
                text_dim=text_dim,
                fusion_dim=fusion_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final fusion layer
        if fusion_output_dim is None:
            fusion_output_dim = visual_dim + text_dim
            
        self.fusion_output = nn.Sequential(
            nn.Linear(visual_dim + text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_output_dim)
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_individual_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the cross-modal transformer.
        
        Args:
            visual_features: Visual features [batch_size, visual_seq_len, visual_dim]
            text_features: Text features [batch_size, text_seq_len, text_dim]
            visual_padding_mask: Visual padding mask [batch_size, visual_seq_len]
            text_padding_mask: Text padding mask [batch_size, text_seq_len]
            return_attention: Whether to return attention weights
            return_individual_features: Whether to return individual modality features
            
        Returns:
            Dictionary containing fused features and optional intermediate outputs
        """
        attention_maps = []
        
        # Process through transformer layers
        for layer in self.layers:
            layer_outputs = layer(
                visual_features,
                text_features,
                visual_padding_mask,
                text_padding_mask,
                return_attention
            )
            
            visual_features = layer_outputs["visual_features"]
            text_features = layer_outputs["text_features"]
            
            if return_attention:
                attention_maps.append({
                    key: val for key, val in layer_outputs.items() 
                    if 'attention' in key
                })
        
        # Get sequence-level representation by averaging tokens
        # For visual: [batch_size, visual_seq_len, visual_dim] -> [batch_size, visual_dim]
        if visual_padding_mask is not None:
            # Only average over non-padding tokens
            visual_rep = (visual_features * (~visual_padding_mask).float().unsqueeze(-1)).sum(1)
            visual_rep = visual_rep / (~visual_padding_mask).float().sum(1, keepdim=True).clamp(min=1e-9)
        else:
            visual_rep = visual_features.mean(dim=1)
            
        # For text: [batch_size, text_seq_len, text_dim] -> [batch_size, text_dim]
        if text_padding_mask is not None:
            # Only average over non-padding tokens
            text_rep = (text_features * (~text_padding_mask).float().unsqueeze(-1)).sum(1)
            text_rep = text_rep / (~text_padding_mask).float().sum(1, keepdim=True).clamp(min=1e-9)
        else:
            text_rep = text_features.mean(dim=1)
        
        # Concatenate modality representations
        # [batch_size, visual_dim + text_dim]
        concat_rep = torch.cat([visual_rep, text_rep], dim=1)
        
        # Apply fusion output layer
        # [batch_size, fusion_output_dim]
        fused_features = self.fusion_output(concat_rep)
        
        # Prepare return dictionary
        result = {"fused_features": fused_features}
        
        # Add optional outputs
        if return_individual_features:
            result.update({
                "visual_features": visual_features,
                "text_features": text_features,
                "visual_representation": visual_rep,
                "text_representation": text_rep
            })
            
        if return_attention and attention_maps:
            result["attention_maps"] = attention_maps
            
        return result