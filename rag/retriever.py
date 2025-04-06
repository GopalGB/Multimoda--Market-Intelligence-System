# rag/retriever.py
import logging
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from tqdm import tqdm
import re
from transformers import AutoModel, AutoTokenizer

# Local imports
from .vector_store import VectorStore
from .reranker import Reranker

logger = logging.getLogger(__name__)

class Retriever:
    """
    Base retriever for RAG system.
    
    This class provides abstract methods for retrieving relevant documents
    to support the RAG (Retrieval-Augmented Generation) pipeline.
    """
    def __init__(self):
        # Statistics tracking
        self.stats = {
            "total_queries": 0,
            "avg_retrieval_time": 0,
            "last_retrieval_time": 0
        }
    
    def search(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        raise NotImplementedError("Subclasses must implement search()")
    
    def batch_search(self, queries: List[str], top_k: int = 10, **kwargs) -> List[List[Dict[str, Any]]]:
        """
        Retrieve relevant documents for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve per query
            
        Returns:
            List of lists of relevant documents
        """
        # Default implementation: search for each query individually
        results = []
        for query in queries:
            results.append(self.search(query, top_k, **kwargs))
        return results

    def update_stats(self, retrieval_time: float) -> None:
        """Update retrieval statistics."""
        self.stats["total_queries"] += 1
        self.stats["last_retrieval_time"] = retrieval_time
        
        # Update moving average
        if self.stats["total_queries"] == 1:
            self.stats["avg_retrieval_time"] = retrieval_time
        else:
            self.stats["avg_retrieval_time"] = (
                0.95 * self.stats["avg_retrieval_time"] + 0.05 * retrieval_time
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return self.stats


class DenseRetriever(Retriever):
    """
    Dense retriever using embedding similarity for document retrieval.
    
    This retriever encodes queries into embeddings and uses vector similarity
    to retrieve relevant documents from a vector store.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store: Optional[VectorStore] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the dense retriever.
        
        Args:
            model_name: Name of the model for encoding queries
            vector_store: VectorStore for document vectors
            cache_dir: Directory for caching models
            device: Device to run the model on
        """
        super().__init__()
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize embedding model and tokenizer
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
        # Initialize vector store
        if vector_store is None:
            # Get embedding dimension from model
            self.vector_store = VectorStore(
                dimension=self.model.config.hidden_size,
                index_type="hnsw",
                metric="cosine"
            )
        else:
            self.vector_store = vector_store
    
    def _initialize_model(self) -> None:
        """Initialize the embedding model and tokenizer."""
        logger.info(f"Loading retrieval model: {self.model_name}")
        
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Move model to device
            self.model.to(self.device)
            
            # Set evaluation mode
            self.model.eval()
            
            logger.info(f"Retrieval model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading retrieval model: {str(e)}")
            raise
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query into an embedding vector.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector
        """
        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token embedding as query representation
        embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
        
        return embeddings
    
    def encode_batch(self, queries: List[str]) -> np.ndarray:
        """
        Encode multiple queries into embedding vectors.
        
        Args:
            queries: List of query strings
            
        Returns:
            Batch of query embedding vectors
        """
        # Tokenize
        inputs = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token embedding as query representation
        embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
        
        return embeddings
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: Optional[Callable[[str, Dict], bool]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            filter_fn: Optional function to filter results
            
        Returns:
            List of relevant documents
        """
        start_time = time.time()
        
        # Encode query to embedding
        query_vector = self.encode_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_vector, k=top_k, filter_fn=filter_fn)
        
        # Update statistics
        retrieval_time = time.time() - start_time
        self.update_stats(retrieval_time)
        
        # Add query to results
        for result in results:
            result["query"] = query
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        filter_fn: Optional[Callable[[str, Dict], bool]] = None,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve relevant documents for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve per query
            filter_fn: Optional function to filter results
            
        Returns:
            List of lists of relevant documents
        """
        start_time = time.time()
        
        # Encode all queries
        query_vectors = self.encode_batch(queries)
        
        # Search vector store for all queries
        all_results = self.vector_store.batch_search(query_vectors, k=top_k, filter_fn=filter_fn)
        
        # Add queries to results
        for i, results in enumerate(all_results):
            for result in results:
                result["query"] = queries[i]
        
        # Update statistics
        retrieval_time = time.time() - start_time
        self.update_stats(retrieval_time / len(queries))
        
        return all_results
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the retriever's vector store.
        
        Args:
            documents: List of documents to add
            embeddings: Optional pre-computed embeddings
            ids: Optional document IDs
            
        Returns:
            List of added document IDs
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if embeddings is None:
            # Would need to extract text and compute embeddings
            raise NotImplementedError("Document embedding not implemented in base retriever")
        
        # Add documents with embeddings to vector store
        return self.vector_store.add(embeddings, documents, ids)


class SparseRetriever(Retriever):
    """
    Sparse retriever using keyword matching for document retrieval.
    
    This retriever uses traditional IR techniques like BM25 to retrieve
    relevant documents based on keyword matching.
    """
    def __init__(
        self,
        algorithm: str = "bm25",
        documents: Optional[List[Dict[str, Any]]] = None,
        text_field: str = "text"
    ):
        """
        Initialize the sparse retriever.
        
        Args:
            algorithm: Retrieval algorithm ("bm25", "tfidf")
            documents: Optional list of documents to index
            text_field: Field containing document text
        """
        super().__init__()
        self.algorithm = algorithm
        self.text_field = text_field
        
        # Initialize index
        if algorithm == "bm25":
            try:
                from rank_bm25 import BM25Okapi
                self.index_cls = BM25Okapi
            except ImportError:
                raise ImportError("rank_bm25 is required for BM25 retrieval. Install with 'pip install rank-bm25'")
        elif algorithm == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.vectorizer = TfidfVectorizer()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Document storage
        self.documents = []
        self.doc_ids = []
        self.index = None
        
        # Build index if documents provided
        if documents:
            self.add_documents(documents)
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the retriever's index.
        
        Args:
            documents: List of documents to add
            ids: Optional document IDs
            
        Returns:
            List of added document IDs
        """
        # Extract text from documents
        texts = []
        for doc in documents:
            text = doc.get(self.text_field, "")
            if not text and "content" in doc:
                text = doc["content"]
            if not text and "body" in doc:
                text = doc["body"]
            texts.append(text)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i+len(self.doc_ids)}" for i in range(len(documents))]
        
        # Store documents and IDs
        self.documents.extend(documents)
        self.doc_ids.extend(ids)
        
        # Build or update index
        if self.algorithm == "bm25":
            # Tokenize texts
            tokenized_texts = [self._tokenize(text) for text in texts]
            # Build index
            self.index = self.index_cls(tokenized_texts)
        elif self.algorithm == "tfidf":
            # Update vectorizer with all texts
            all_texts = [doc.get(self.text_field, "") for doc in self.documents]
            self.vectorizer.fit(all_texts)
            # Transform to TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.transform(all_texts)
        
        return ids
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.documents:
            return []
        
        start_time = time.time()
        
        if self.algorithm == "bm25":
            # Tokenize query
            query_tokens = self._tokenize(query)
            
            # Get BM25 scores
            scores = self.index.get_scores(query_tokens)
            
            # Get top-k document indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Prepare results
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append({
                        "id": self.doc_ids[idx],
                        "score": float(scores[idx]),
                        "document": self.documents[idx],
                        "query": query
                    })
        
        elif self.algorithm == "tfidf":
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # Get top-k document indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Prepare results
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    results.append({
                        "id": self.doc_ids[idx],
                        "score": float(similarities[idx]),
                        "document": self.documents[idx],
                        "query": query
                    })
        
        # Update statistics
        retrieval_time = time.time() - start_time
        self.update_stats(retrieval_time)
        
        return results


class HybridRetriever(Retriever):
    """
    Hybrid retriever combining dense and sparse retrieval methods.
    
    This retriever combines the strengths of both vector similarity (dense)
    and keyword matching (sparse) approaches for more robust retrieval.
    """
    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        reranker: Optional[Reranker] = None,
        fusion_method: str = "interpolation",  # "interpolation", "reciprocal_rank_fusion"
        alpha: float = 0.5  # Weight between dense (0) and sparse (1)
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            dense_retriever: Dense retrieval component
            sparse_retriever: Sparse retrieval component
            reranker: Optional reranker for results
            fusion_method: Method for combining results
            alpha: Weight between dense and sparse results
        """
        super().__init__()
        
        # Initialize retrievers if not provided
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or SparseRetriever()
        
        self.reranker = reranker
        self.fusion_method = fusion_method
        self.alpha = alpha
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        embeddings: Optional[np.ndarray] = None
    ) -> List[str]:
        """
        Add documents to both retrievers.
        
        Args:
            documents: List of documents to add
            ids: Optional document IDs
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of added document IDs
        """
        # Add to sparse retriever
        sparse_ids = self.sparse_retriever.add_documents(documents, ids)
        
        # Add to dense retriever if embeddings are provided
        if embeddings is not None:
            self.dense_retriever.add_documents(documents, embeddings, sparse_ids)
        
        return sparse_ids
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_dense: bool = True,
        use_sparse: bool = True,
        rerank: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid approach.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            use_dense: Whether to use dense retrieval
            use_sparse: Whether to use sparse retrieval
            rerank: Whether to rerank results
            
        Returns:
            List of relevant documents
        """
        start_time = time.time()
        
        # Get more results than needed for fusion
        retrieve_k = top_k * 2
        
        # Get results from both retrievers
        dense_results = []
        sparse_results = []
        
        if use_dense:
            dense_results = self.dense_retriever.search(query, retrieve_k)
        
        if use_sparse:
            sparse_results = self.sparse_retriever.search(query, retrieve_k)
        
        # Combine results based on fusion method
        if self.fusion_method == "interpolation":
            results = self._interpolation_fusion(
                dense_results, sparse_results, alpha=self.alpha
            )
        elif self.fusion_method == "reciprocal_rank_fusion":
            results = self._reciprocal_rank_fusion(
                dense_results, sparse_results
            )
        else:
            # Default to dense results if both are available, otherwise use what we have
            results = dense_results if dense_results else sparse_results
        
        # Limit to top_k
        results = results[:top_k]
        
        # Rerank if requested and reranker is available
        if rerank and self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=top_k)
        
        # Update statistics
        retrieval_time = time.time() - start_time
        self.update_stats(retrieval_time)
        
        return results
    
    def _interpolation_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Combine results using score interpolation.
        
        Args:
            dense_results: Results from dense retriever
            sparse_results: Results from sparse retriever
            alpha: Weight between dense (0) and sparse (1) scores
            
        Returns:
            Combined results
        """
        # Create a combined dictionary of all results
        combined = {}
        
        # Normalize scores within each result set
        def normalize_scores(results):
            if not results:
                return {}
            
            scores = [r["score"] for r in results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            normalized = {}
            for r in results:
                doc_id = r["id"]
                if score_range > 0:
                    normalized_score = (r["score"] - min_score) / score_range
                else:
                    normalized_score = 1.0  # If all scores are the same
                normalized[doc_id] = {
                    "score": normalized_score,
                    "document": r["document"],
                    "query": r["query"]
                }
            
            return normalized
        
        # Normalize scores
        dense_normalized = normalize_scores(dense_results)
        sparse_normalized = normalize_scores(sparse_results)
        
        # Combine results
        for doc_id in set(list(dense_normalized.keys()) + list(sparse_normalized.keys())):
            dense_score = dense_normalized.get(doc_id, {}).get("score", 0.0)
            sparse_score = sparse_normalized.get(doc_id, {}).get("score", 0.0)
            
            # Interpolate scores
            combined_score = (1 - alpha) * dense_score + alpha * sparse_score
            
            # Get document from whichever retriever found it
            document = (
                dense_normalized.get(doc_id, {}).get("document") or
                sparse_normalized.get(doc_id, {}).get("document")
            )
            
            # Get query
            query = (
                dense_normalized.get(doc_id, {}).get("query") or
                sparse_normalized.get(doc_id, {}).get("query")
            )
            
            combined[doc_id] = {
                "id": doc_id,
                "score": combined_score,
                "document": document,
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "query": query
            }
        
        # Convert to list and sort by score
        results = list(combined.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score for a document is the sum of 1/(rank + k) for all retrievers.
        
        Args:
            dense_results: Results from dense retriever
            sparse_results: Results from sparse retriever
            k: Constant to avoid high impact of high rankings
            
        Returns:
            Combined results
        """
        # Track document scores and info
        doc_scores = {}
        doc_info = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            doc_id = result["id"]
            # RRF formula: 1/(rank + k)
            score = 1.0 / (rank + k)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            doc_info[doc_id] = {
                "document": result["document"],
                "dense_score": result.get("score", 0.0),
                "sparse_score": 0.0,
                "query": result.get("query", "")
            }
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = result["id"]
            # RRF formula: 1/(rank + k)
            score = 1.0 / (rank + k)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            
            if doc_id in doc_info:
                # Update existing entry
                doc_info[doc_id]["sparse_score"] = result.get("score", 0.0)
            else:
                # Create new entry
                doc_info[doc_id] = {
                    "document": result["document"],
                    "dense_score": 0.0,
                    "sparse_score": result.get("score", 0.0),
                    "query": result.get("query", "")
                }
        
        # Combine results
        results = []
        for doc_id, score in doc_scores.items():
            info = doc_info[doc_id]
            results.append({
                "id": doc_id,
                "score": score,
                "document": info["document"],
                "dense_score": info["dense_score"],
                "sparse_score": info["sparse_score"],
                "query": info["query"]
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results