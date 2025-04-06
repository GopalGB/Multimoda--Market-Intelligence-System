# rag/reranker.py
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Reranker:
    """
    Reranker component for the RAG system.
    
    This class provides methods to:
    - Rerank retrieved documents using relevance scoring
    - Implement cross-encoder models for better result quality
    - Support different reranking strategies
    - Filter and diversify results
    
    It's designed to improve the quality of retrieved passages by using
    more sophisticated relevance modeling than the initial retrieval.
    """
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
            device: Device to run the model on ('cpu', 'cuda', etc.)
            cache_dir: Directory for caching models
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
        # Track usage statistics
        self.stats = {
            "total_reranked": 0,
            "avg_rerank_time": 0,
            "last_rerank_time": 0
        }
    
    def _initialize_model(self) -> None:
        """Initialize the reranking model and tokenizer."""
        logger.info(f"Loading reranker model: {self.model_name}")
        
        try:
            # Load cross-encoder model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Move model to device
            self.model.to(self.device)
            
            # Set evaluation mode
            self.model.eval()
            
            logger.info(f"Reranker model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading reranker model: {str(e)}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        document_content_key: str = "text",
        return_scores: bool = True,
        threshold: Optional[float] = None,
        normalize_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents based on relevance to query.
        
        Args:
            query: Query string
            documents: List of document dictionaries
            top_k: Number of top documents to return (None for all)
            document_content_key: Key to access document content
            return_scores: Whether to include scores in results
            threshold: Minimum score threshold for inclusion
            normalize_scores: Whether to normalize scores to [0-1] range
            
        Returns:
            List of reranked documents
        """
        if not documents:
            return []
        
        start_time = time.time()
        
        # Extract text content from documents
        doc_texts = []
        for doc in documents:
            document = doc.get("document", doc)  # Handle different document formats
            
            # Get text content
            if document_content_key in document:
                doc_texts.append(str(document[document_content_key]))
            elif "content" in document:
                doc_texts.append(str(document["content"]))
            elif "text" in document:
                doc_texts.append(str(document["text"]))
            elif "body" in document:
                doc_texts.append(str(document["body"]))
            else:
                # Try to convert document to string as fallback
                doc_texts.append(str(document))
        
        # Rerank in batches
        all_scores = []
        for i in range(0, len(doc_texts), self.batch_size):
            batch_texts = doc_texts[i:i+self.batch_size]
            batch_scores = self._score_batch(query, batch_texts)
            all_scores.extend(batch_scores)
        
        # Normalize scores if requested
        if normalize_scores:
            # Min-max normalization to [0, 1]
            if len(all_scores) > 0:
                min_score = min(all_scores)
                max_score = max(all_scores)
                score_range = max_score - min_score
                if score_range > 0:
                    all_scores = [(s - min_score) / score_range for s in all_scores]
                else:
                    all_scores = [1.0 for _ in all_scores]
        
        # Filter by threshold if provided
        filtered_indices = list(range(len(all_scores)))
        if threshold is not None:
            filtered_indices = [i for i, score in enumerate(all_scores) if score >= threshold]
        
        # Create reranked results
        reranked = []
        for idx in filtered_indices:
            result = documents[idx].copy()
            if return_scores:
                result["reranker_score"] = float(all_scores[idx])
            reranked.append(result)
        
        # Sort by score
        reranked.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)
        
        # Limit to top_k if specified
        if top_k is not None:
            reranked = reranked[:top_k]
        
        # Update statistics
        self.stats["total_reranked"] += len(documents)
        self.stats["last_rerank_time"] = time.time() - start_time
        if self.stats.get("avg_rerank_time", 0) == 0:
            self.stats["avg_rerank_time"] = self.stats["last_rerank_time"]
        else:
            self.stats["avg_rerank_time"] = 0.95 * self.stats["avg_rerank_time"] + 0.05 * self.stats["last_rerank_time"]
        
        return reranked
    
    def _score_batch(
        self,
        query: str,
        texts: List[str]
    ) -> List[float]:
        """
        Score a batch of query-document pairs.
        
        Args:
            query: Query string
            texts: List of document texts
            
        Returns:
            List of relevance scores
        """
        # Prepare input for cross-encoder
        features = self.tokenizer(
            [query] * len(texts),
            texts,
            padding=True,
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Compute relevance scores
        with torch.no_grad():
            outputs = self.model(**features)
            scores = outputs.logits.cpu().numpy()
            
            # Handle different model output formats
            if scores.shape[1] == 1:
                # Regression model
                scores = scores.flatten()
            else:
                # Classification model - use positive class probability or convert to score
                if scores.shape[1] == 2:
                    scores = torch.nn.functional.softmax(torch.tensor(scores), dim=1).numpy()[:, 1]
                else:
                    scores = scores.flatten()
        
        return scores.tolist()
    
    def reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[Dict[str, Any]]],
        k: float = 60.0,
        id_key: str = "id",
        score_key: str = "score"
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.
        
        RRF assigns a score to each document based on its ranks in different lists:
        RRF(d) = sum_i 1/(k + rank_i(d))
        
        Args:
            ranked_lists: List of ranked document lists
            k: Constant to avoid high impact of high rankings
            id_key: Key to identify documents
            score_key: Key for original scores
            
        Returns:
            Fused ranked list
        """
        if not ranked_lists:
            return []
        
        # Track document scores by ID
        doc_scores = {}
        all_docs = {}
        
        # Process each ranked list
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                # Extract document ID
                doc_id = doc.get(id_key, f"doc_{id(doc)}")
                
                # Store document if not seen before
                if doc_id not in all_docs:
                    all_docs[doc_id] = doc
                
                # Add reciprocal rank score
                rrf_score = 1.0 / (k + rank)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
        
        # Create final ranked list
        results = []
        for doc_id, score in doc_scores.items():
            doc = all_docs[doc_id].copy()
            doc["fusion_score"] = score
            results.append(doc)
        
        # Sort by fusion score
        results.sort(key=lambda x: x["fusion_score"], reverse=True)
        
        return results
    
    def diversify(
        self,
        documents: List[Dict[str, Any]],
        embedding_key: str = "embedding",
        diversity_factor: float = 0.5,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents to increase diversity using maximal marginal relevance.
        
        Args:
            documents: List of documents with embeddings
            embedding_key: Key to access document embeddings
            diversity_factor: Trade-off between relevance and diversity (0-1)
            top_k: Number of documents to return
            
        Returns:
            Diversified list of documents
        """
        if not documents:
            return []
        
        # Set default top_k if not specified
        if top_k is None:
            top_k = len(documents)
        top_k = min(top_k, len(documents))
        
        # Extract embeddings and relevance scores
        embeddings = []
        for doc in documents:
            # Get embedding
            emb = doc.get(embedding_key, None)
            if emb is None and "document" in doc:
                emb = doc["document"].get(embedding_key, None)
            
            if emb is not None:
                # Convert to numpy array if needed
                if isinstance(emb, list):
                    emb = np.array(emb)
                embeddings.append(emb)
            else:
                # Use zeros as fallback
                dim = 768  # Default dimension
                if embeddings and len(embeddings) > 0:
                    dim = embeddings[0].shape[0]
                embeddings.append(np.zeros(dim))
        
        # Prepare scores - use reranker_score if available, otherwise initial score
        scores = [doc.get("reranker_score", doc.get("score", 1.0)) for doc in documents]
        
        # Apply maximal marginal relevance
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        # Select the highest scoring document first
        best_idx = max(remaining_indices, key=lambda i: scores[i])
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # Select remaining documents
        while len(selected_indices) < top_k and remaining_indices:
            best_score = -float("inf")
            best_idx = -1
            
            for idx in remaining_indices:
                # Relevance score
                relevance = scores[idx]
                
                # Diversity score (negative similarity to already selected documents)
                similarity_scores = []
                for sel_idx in selected_indices:
                    sim = np.dot(embeddings[idx], embeddings[sel_idx]) / (
                        max(np.linalg.norm(embeddings[idx]) * np.linalg.norm(embeddings[sel_idx]), 1e-8)
                    )
                    similarity_scores.append(sim)
                
                # Maximum similarity to any already selected document
                max_similarity = max(similarity_scores) if similarity_scores else 0
                
                # Combined score with diversity factor
                combined_score = (1 - diversity_factor) * relevance - diversity_factor * max_similarity
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        # Create diversified result list
        diversified = [documents[idx] for idx in selected_indices]
        
        return diversified
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the reranker.
        
        Returns:
            Dictionary with statistics
        """
        return self.stats
    
    def batch_rerank(
        self,
        queries: List[str],
        document_lists: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
        document_content_key: str = "text",
        return_scores: bool = True,
        threshold: Optional[float] = None,
        normalize_scores: bool = True,
        show_progress: bool = False
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple queries and document lists.
        
        Args:
            queries: List of query strings
            document_lists: List of document lists (one per query)
            top_k: Number of top documents to return per query
            document_content_key: Key to access document content
            return_scores: Whether to include scores in results
            threshold: Minimum score threshold for inclusion
            normalize_scores: Whether to normalize scores
            show_progress: Whether to show progress bar
            
        Returns:
            List of reranked document lists
        """
        if len(queries) != len(document_lists):
            raise ValueError("Number of queries must match number of document lists")
        
        # Rerank each query-documents pair
        reranked_lists = []
        
        iterator = range(len(queries))
        if show_progress:
            iterator = tqdm(iterator, desc="Reranking", total=len(queries))
        
        for i in iterator:
            reranked = self.rerank(
                queries[i],
                document_lists[i],
                top_k=top_k,
                document_content_key=document_content_key,
                return_scores=return_scores,
                threshold=threshold,
                normalize_scores=normalize_scores
            )
            reranked_lists.append(reranked)
        
        return reranked_lists

class ColBERTReranker(Reranker):
    """
    Reranker implementation using ColBERT for more efficient reranking.
    Provides a balance between efficiency of retrieval and accuracy of reranking.
    """
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 256
    ):
        """
        Initialize the ColBERT reranker.
        
        Args:
            model_name: Name of the ColBERT model to use
            device: Device to run the model on ('cpu', 'cuda', etc.)
            cache_dir: Directory for caching models
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        """
        # Initialize with base parameters
        super().__init__(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            batch_size=batch_size,
            max_length=max_length
        )
        
        # ColBERT specific initialization will be handled in _initialize_model
    
    def _initialize_model(self) -> None:
        """Initialize the ColBERT model and tokenizer."""
        logger.info("ColBERT reranker not fully implemented - using fallback cross-encoder")
        
        # Fall back to cross-encoder for now
        # In a full implementation, this would load a ColBERT model
        fallback_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            fallback_model,
            cache_dir=self.cache_dir
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            fallback_model,
            cache_dir=self.cache_dir
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Set evaluation mode
        self.model.eval()
        
        logger.info(f"Using fallback reranker model: {fallback_model}")
    
    def _score_batch(
        self,
        query: str,
        texts: List[str]
    ) -> List[float]:
        """
        Score a batch of query-document pairs using ColBERT.
        
        Args:
            query: Query string
            texts: List of document texts
            
        Returns:
            List of relevance scores
        """
        # In a full implementation, this would use ColBERT's late interaction scoring
        # For now, use the same scoring as the base class
        return super()._score_batch(query, texts)