# models/text/roberta_model.py
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from transformers import RobertaModel, RobertaTokenizer

class RoBERTaWrapper(nn.Module):
    """
    Wrapper for the RoBERTa model for text analysis.
    """
    def __init__(
        self,
        model_name: str = "roberta-base",
        device: Optional[str] = None,
        pooling_strategy: str = "cls"  # cls, mean, max
    ):
        """
        Initialize the RoBERTa wrapper.
        
        Args:
            model_name: Name of the RoBERTa model to use
            device: Device to run the model on (cpu, cuda, auto)
            pooling_strategy: Strategy for sentence embedding pooling
        """
        super().__init__()
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set pooling strategy
        self.pooling_strategy = pooling_strategy
        
        # Load model and tokenizer
        self.model = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Move model to device
        self.model.to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        return_hidden_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode text into embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Text embeddings and optionally hidden states
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=return_hidden_states)
        
        # Apply pooling strategy
        if self.pooling_strategy == "cls":
            # Use CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "mean":
            # Mean pooling - average all tokens
            attention_mask = inputs["attention_mask"]
            embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooling_strategy == "max":
            # Max pooling - take max over tokens
            embeddings = outputs.last_hidden_state.max(dim=1)[0]
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
        
        # Normalize if requested
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if return_hidden_states:
            return embeddings, outputs.hidden_states
        
        return embeddings
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling over token embeddings.
        
        Args:
            token_embeddings: Token embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled embeddings [batch_size, hidden_size]
        """
        # Expand mask to same dimension as embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings weighted by mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum mask to get actual token count
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Average embeddings over tokens
        pooled = sum_embeddings / sum_mask
        
        return pooled
    
    def calculate_similarity(
        self,
        texts1: Union[str, List[str]],
        texts2: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Calculate cosine similarity between two sets of texts.
        
        Args:
            texts1: First set of texts
            texts2: Second set of texts
            
        Returns:
            Similarity matrix
        """
        # Encode texts
        embeddings1 = self.encode_text(texts1)
        embeddings2 = self.encode_text(texts2)
        
        # Calculate similarity
        similarity = torch.matmul(embeddings1, embeddings2.t())
        
        return similarity
    
    def analyze_sentiment(
        self,
        texts: Union[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Define sentiment prompts
        sentiment_prompts = [
            "This text expresses a very negative sentiment.",
            "This text expresses a negative sentiment.",
            "This text expresses a neutral sentiment.",
            "This text expresses a positive sentiment.",
            "This text expresses a very positive sentiment."
        ]
        
        # Convert single text to list
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Calculate similarity with sentiment prompts
        similarities = []
        
        for text in texts:
            text_emb = self.encode_text(text)
            prompt_emb = self.encode_text(sentiment_prompts)
            
            # Calculate similarity with each prompt
            sim = torch.matmul(text_emb, prompt_emb.t())
            sim = F.softmax(sim * 10, dim=1)  # Scale for sharper distribution
            
            similarities.append(sim[0])
        
        # Convert to sentiment scores (-1 to 1)
        sentiment_weights = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=self.device)
        sentiment_scores = [torch.sum(sim * sentiment_weights).item() for sim in similarities]
        
        # Create detailed results
        results = []
        for i, text in enumerate(texts):
            # Get sentiment label based on score
            score = sentiment_scores[i]
            if score <= -0.6:
                label = "very negative"
            elif score <= -0.2:
                label = "negative"
            elif score <= 0.2:
                label = "neutral"
            elif score <= 0.6:
                label = "positive"
            else:
                label = "very positive"
            
            # Create detailed breakdown
            breakdown = {
                prompt: similarities[i][j].item()
                for j, prompt in enumerate([
                    "very negative", "negative", "neutral", "positive", "very positive"
                ])
            }
            
            results.append({
                "score": score,
                "label": label,
                "breakdown": breakdown
            })
        
        # Return single result for single input
        if single_input:
            return results[0]
        
        return {"batch_results": results}
    
    def extract_keywords(
        self,
        texts: Union[str, List[str]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Extract keywords from texts using attention weights.
        
        Args:
            texts: Single text or list of texts
            top_k: Number of top keywords to extract
            
        Returns:
            Dictionary with keyword extraction results
        """
        # Convert single text to list
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Tokenize
        tokenized = [self.tokenizer.tokenize(text) for text in texts]
        
        # Encode with hidden states
        _, hidden_states = self.encode_text(texts, return_hidden_states=True)
        
        # Get attention from last layer
        with torch.no_grad():
            # Re-encode to get attention weights
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            outputs = self.model(**inputs, output_attentions=True)
            
            # Get attention weights from last layer (average over heads)
            # [batch_size, num_heads, seq_len, seq_len]
            attentions = outputs.attentions[-1].mean(dim=1)
        
        # Extract keywords based on attention
        results = []
        
        for i, text in enumerate(texts):
            # Get attention for CLS token
            cls_attention = attentions[i, 0, :].cpu().numpy()
            
            # Get token IDs and attention mask
            token_ids = inputs.input_ids[i].cpu().numpy()
            mask = inputs.attention_mask[i].cpu().numpy()
            
            # Convert to tokens
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            
            # Filter special tokens and apply mask
            token_attention = []
            for j, token in enumerate(tokens):
                if mask[j] == 1 and not token.startswith("<") and not token.startswith("Ġ"):
                    token_attention.append((token, cls_attention[j]))
            
            # Sort by attention score
            token_attention.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k
            top_keywords = [
                {"keyword": token, "score": float(score)}
                for token, score in token_attention[:top_k]
            ]
            
            results.append({
                "keywords": top_keywords
            })
        
        # Return single result for single input
        if single_input:
            return results[0]
        
        return {"batch_results": results}