# rag/vector_store.py
import os
import json
import numpy as np
import faiss
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import torch
import pickle
from pathlib import Path
import datetime
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector storage system for efficient similarity search.
    
    This class provides methods to:
    - Store and retrieve document embeddings
    - Perform similarity search with exact and approximate methods
    - Support hybrid search (combining dense and sparse retrieval)
    - Manage document metadata alongside vectors
    - Persist and load vector indices
    
    It's optimized for the retrieval component of the RAG system,
    supporting both in-memory and disk-based storage options.
    """
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "flat",  # "flat", "hnsw", "ivf", "pq"
        metric: str = "cosine",  # "cosine", "l2", "ip" (inner product)
        storage_dir: Optional[str] = None,
        index_name: str = "default",
        device: Optional[str] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimensionality of vectors
            index_type: Type of FAISS index to use
            metric: Distance metric for similarity
            storage_dir: Directory for persisting indices (None for in-memory only)
            index_name: Name of the index
            device: Device to use for computations (None for auto-detect)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric
        self.index_name = index_name
        self.storage_dir = storage_dir
        
        # Create storage directory if provided
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
        
        # Device for computations
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Convert metric string to FAISS metric type
        if metric == "cosine":
            self.faiss_metric = faiss.METRIC_INNER_PRODUCT
            self.normalize = True
        elif metric == "l2":
            self.faiss_metric = faiss.METRIC_L2
            self.normalize = False
        elif metric == "ip":
            self.faiss_metric = faiss.METRIC_INNER_PRODUCT
            self.normalize = False
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Initialize index
        self.index = self._create_index()
        
        # Document storage
        self.doc_ids = []
        self.documents = {}
        self.id_to_index = {}  # Mapping from doc_id to index in the FAISS store
        
        # Metadata storage
        self.metadata = {}
        
        # Index statistics
        self.stats = {
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "num_vectors": 0,
            "dimension": dimension,
            "index_type": index_type,
            "metric": metric
        }
    
    def _create_index(self) -> faiss.Index:
        """
        Create a FAISS index based on configuration.
        
        Returns:
            FAISS index
        """
        # Create base index based on metric
        if self.index_type == "flat":
            # Exact search (slowest but most accurate)
            if self.faiss_metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(self.dimension)
            else:  # L2 distance
                index = faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World (fast and accurate)
            # M parameter controls graph connectivity (usually 16-64)
            # efConstruction controls index quality (usually 40-400)
            m = 32
            ef_construction = 200
            
            if self.faiss_metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexHNSWFlat(self.dimension, m, self.faiss_metric)
            else:  # L2 distance
                index = faiss.IndexHNSWFlat(self.dimension, m, self.faiss_metric)
                
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = 128  # Controls search accuracy vs speed
        
        elif self.index_type == "ivf":
            # Inverted File Index (faster but less accurate)
            # nlist controls number of partitions (more = better accuracy but slower)
            nlist = 100  # rule of thumb: sqrt(num_vectors) but adjust based on size
            quantizer = faiss.IndexFlatL2(self.dimension)
            
            if self.faiss_metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, self.faiss_metric)
            else:  # L2 distance
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, self.faiss_metric)
                
            # Needs training with sample vectors before adding real vectors
            # We'll defer training until we have vectors
        
        elif self.index_type == "pq":
            # Product Quantization (much faster, much smaller, less accurate)
            # m controls number of subquantizers (more = better accuracy but larger index)
            # bits_per_code controls precision per subquantizer (more = better accuracy)
            m = 8  # number of subquantizers
            bits_per_code = 8  # bits per subquantizer (8 = 1 byte)
            
            if self.faiss_metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexPQ(self.dimension, m, bits_per_code, self.faiss_metric)
            else:  # L2 distance
                index = faiss.IndexPQ(self.dimension, m, bits_per_code, self.faiss_metric)
                
            # Needs training with sample vectors before adding real vectors
            # We'll defer training until we have vectors
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Move to GPU if available and appropriate
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                # Use first GPU by default
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info(f"Using GPU for FAISS index {self.index_name}")
            except Exception as e:
                logger.warning(f"Failed to use GPU for FAISS index: {str(e)}")
        
        return index
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity.
        
        Args:
            vectors: Input vectors [n, dimension]
            
        Returns:
            Normalized vectors
        """
        if not self.normalize:
            return vectors
        
        # Calculate L2 norm
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        # Normalize
        return vectors / norms
    
    def add(
        self,
        vectors: np.ndarray,
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add vectors and documents to the store.
        
        Args:
            vectors: Embedding vectors [n, dimension]
            documents: List of document dictionaries
            ids: Optional list of document IDs (auto-generated if None)
            
        Returns:
            List of document IDs
        """
        # Check inputs
        if len(vectors) != len(documents):
            raise ValueError(f"Number of vectors ({len(vectors)}) must match number of documents ({len(documents)})")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i+len(self.doc_ids)}" for i in range(len(vectors))]
        elif len(ids) != len(vectors):
            raise ValueError(f"Number of IDs ({len(ids)}) must match number of vectors ({len(vectors)})")
        
        # Check if index requires training
        if self.index_type in ["ivf", "pq"] and not hasattr(self.index, "ntotal"):
            # Train the index if we have vectors
            self.index.train(vectors)
        
        # Normalize vectors if using cosine similarity
        if self.normalize:
            vectors = self._normalize_vectors(vectors)
        
        # Add vectors to the index
        start_idx = len(self.doc_ids)
        self.index.add(vectors)
        
        # Update document storage
        for i, (doc_id, doc) in enumerate(zip(ids, documents)):
            idx = start_idx + i
            self.doc_ids.append(doc_id)
            self.documents[doc_id] = doc
            self.id_to_index[doc_id] = idx
            
            # Extract metadata if present
            if "metadata" in doc:
                self.metadata[doc_id] = doc["metadata"]
            else:
                self.metadata[doc_id] = {}
        
        # Update statistics
        self.stats["num_vectors"] += len(vectors)
        self.stats["updated_at"] = datetime.datetime.now().isoformat()
        
        return ids
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_fn: Optional[Callable[[str, Dict], bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector [dimension]
            k: Number of results to return
            filter_fn: Optional function to filter results by metadata
            
        Returns:
            List of dictionaries with search results
        """
        # Ensure query vector has correct shape
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize query vector if using cosine similarity
        if self.normalize:
            query_vector = self._normalize_vectors(query_vector)
        
        # If we need to filter results, retrieve more than k to account for filtered items
        num_to_retrieve = k * 3 if filter_fn else k
        # Limit to available documents
        num_to_retrieve = min(num_to_retrieve, len(self.doc_ids))
        
        if num_to_retrieve == 0:
            return []
        
        # Perform search
        distances, indices = self.index.search(query_vector, num_to_retrieve)
        
        # Prepare results
        results = []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # Skip invalid indices (can happen with approximate search)
            if idx < 0 or idx >= len(self.doc_ids):
                continue
                
            doc_id = self.doc_ids[idx]
            document = self.documents[doc_id]
            
            # Apply filter if provided
            if filter_fn and not filter_fn(doc_id, document):
                continue
            
            # Adjust score based on metric type
            if self.faiss_metric == faiss.METRIC_INNER_PRODUCT:
                # For inner product, higher is better (already normalized for cosine)
                score = float(distance)
            else:  # L2 distance
                # For L2 distance, lower is better, convert to a similarity score
                score = float(1.0 / (1.0 + distance))
            
            # Add to results
            results.append({
                "id": doc_id,
                "score": score,
                "document": document,
                "metadata": self.metadata.get(doc_id, {})
            })
            
            # Stop when we have enough results
            if len(results) >= k:
                break
        
        return results
    
    def batch_search(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        filter_fn: Optional[Callable[[str, Dict], bool]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar vectors in batch.
        
        Args:
            query_vectors: Query vectors [n, dimension]
            k: Number of results to return per query
            filter_fn: Optional function to filter results by metadata
            
        Returns:
            List of lists of dictionaries with search results
        """
        # Normalize query vectors if using cosine similarity
        if self.normalize:
            query_vectors = self._normalize_vectors(query_vectors)
        
        # If we need to filter results, retrieve more than k to account for filtered items
        num_to_retrieve = k * 3 if filter_fn else k
        # Limit to available documents
        num_to_retrieve = min(num_to_retrieve, len(self.doc_ids))
        
        if num_to_retrieve == 0:
            return [[] for _ in range(len(query_vectors))]
        
        # Perform search
        distances, indices = self.index.search(query_vectors, num_to_retrieve)
        
        # Prepare results
        batch_results = []
        
        for query_idx in range(len(query_vectors)):
            results = []
            
            for i, (distance, idx) in enumerate(zip(distances[query_idx], indices[query_idx])):
                # Skip invalid indices
                if idx < 0 or idx >= len(self.doc_ids):
                    continue
                    
                doc_id = self.doc_ids[idx]
                document = self.documents[doc_id]
                
                # Apply filter if provided
                if filter_fn and not filter_fn(doc_id, document):
                    continue
                
                # Adjust score based on metric type
                if self.faiss_metric == faiss.METRIC_INNER_PRODUCT:
                    score = float(distance)
                else:  # L2 distance
                    score = float(1.0 / (1.0 + distance))
                
                # Add to results
                results.append({
                    "id": doc_id,
                    "score": score,
                    "document": document,
                    "metadata": self.metadata.get(doc_id, {})
                })
                
                # Stop when we have enough results
                if len(results) >= k:
                    break
            
            batch_results.append(results)
        
        return batch_results
    
    def get(
        self,
        doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document or None if not found
        """
        if doc_id in self.documents:
            return {
                "id": doc_id,
                "document": self.documents[doc_id],
                "metadata": self.metadata.get(doc_id, {})
            }
        return None
    
    def update(
        self,
        doc_id: str,
        vector: Optional[np.ndarray] = None,
        document: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document and/or its vector.
        
        Args:
            doc_id: Document ID
            vector: New vector (None to keep existing)
            document: New document (None to keep existing)
            
        Returns:
            True if update successful
        """
        if doc_id not in self.id_to_index:
            return False
        
        idx = self.id_to_index[doc_id]
        
        # Update vector if provided
        if vector is not None:
            # For most FAISS indices, we can't update individual vectors
            # Instead, we need to remove and re-add all vectors
            # This is an expensive operation, so we'll just warn about it
            logger.warning(f"Updating individual vectors is not fully supported. "
                          f"The document will be updated but the vector remains unchanged.")
        
        # Update document if provided
        if document is not None:
            self.documents[doc_id] = document
            
            # Update metadata if present
            if "metadata" in document:
                self.metadata[doc_id] = document["metadata"]
        
        # Update statistics
        self.stats["updated_at"] = datetime.datetime.now().isoformat()
        
        return True
    
    def delete(
        self,
        doc_id: str
    ) -> bool:
        """
        Delete a document from the store.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deletion successful
        """
        if doc_id not in self.id_to_index:
            return False
        
        # For most FAISS indices, we can't remove individual vectors
        # Instead, we would need to rebuild the index
        # This is an expensive operation, so we'll just mark as deleted
        
        # Mark document as deleted
        idx = self.id_to_index[doc_id]
        self.documents[doc_id] = {"_deleted": True}
        self.metadata[doc_id] = {"_deleted": True}
        
        # Update statistics
        self.stats["updated_at"] = datetime.datetime.now().isoformat()
        # We don't decrease num_vectors because the vector is still in the index
        
        return True
    
    def save(
        self,
        directory: Optional[str] = None
    ) -> str:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save to (None to use storage_dir)
            
        Returns:
            Path to saved index
        """
        save_dir = directory or self.storage_dir
        
        if not save_dir:
            raise ValueError("No storage directory specified")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the index
        index_path = os.path.join(save_dir, f"{self.index_name}.index")
        
        # Convert to CPU index if on GPU
        if hasattr(self.index, "index") and hasattr(self.index.index, "d"):
            # This is a GPU index, move to CPU
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        else:
            index_to_save = self.index
            
        faiss.write_index(index_to_save, index_path)
        
        # Save document metadata
        docs_path = os.path.join(save_dir, f"{self.index_name}_docs.pkl")
        with open(docs_path, 'wb') as f:
            pickle.dump({
                "doc_ids": self.doc_ids,
                "documents": self.documents,
                "id_to_index": self.id_to_index,
                "metadata": self.metadata
            }, f)
        
        # Save statistics
        stats_path = os.path.join(save_dir, f"{self.index_name}_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        return index_path
    
    @classmethod
    def load(
        cls,
        directory: str,
        index_name: str = "default",
        device: Optional[str] = None
    ) -> 'VectorStore':
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory to load from
            index_name: Name of the index
            device: Device to use (None for auto-detect)
            
        Returns:
            Loaded VectorStore
        """
        # Check if files exist
        index_path = os.path.join(directory, f"{index_name}.index")
        docs_path = os.path.join(directory, f"{index_name}_docs.pkl")
        stats_path = os.path.join(directory, f"{index_name}_stats.json")
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError(f"Index files not found in {directory}")
        
        # Load statistics to initialize with correct parameters
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                
            dimension = stats.get("dimension", 768)
            index_type = stats.get("index_type", "flat")
            metric = stats.get("metric", "cosine")
        else:
            # Default parameters if stats not found
            dimension = 768
            index_type = "flat"
            metric = "cosine"
        
        # Create instance
        vector_store = cls(
            dimension=dimension,
            index_type=index_type,
            metric=metric,
            storage_dir=directory,
            index_name=index_name,
            device=device
        )
        
        # Load index
        index = faiss.read_index(index_path)
        
        # Move to GPU if requested
        if device == "cuda" or (device is None and torch.cuda.is_available()):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info(f"Using GPU for loaded FAISS index {index_name}")
            except Exception as e:
                logger.warning(f"Failed to use GPU for loaded FAISS index: {str(e)}")
        
        vector_store.index = index
        
        # Load documents
        with open(docs_path, 'rb') as f:
            docs_data = pickle.load(f)
            
        vector_store.doc_ids = docs_data["doc_ids"]
        vector_store.documents = docs_data["documents"]
        vector_store.id_to_index = docs_data["id_to_index"]
        vector_store.metadata = docs_data.get("metadata", {})
        
        # Load statistics
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                vector_store.stats = json.load(f)
        
        return vector_store
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        # Update statistics
        self.stats["num_vectors"] = len(self.doc_ids)
        self.stats["num_active"] = sum(1 for doc_id in self.doc_ids
                                      if doc_id in self.documents and 
                                      not self.documents[doc_id].get("_deleted", False))
        
        return self.stats
    
    def clear(self) -> None:
        """
        Clear the vector store.
        """
        # Re-initialize the index
        self.index = self._create_index()
        
        # Clear document storage
        self.doc_ids = []
        self.documents = {}
        self.id_to_index = {}
        self.metadata = {}
        
        # Reset statistics
        self.stats["num_vectors"] = 0
        self.stats["updated_at"] = datetime.datetime.now().isoformat()
    
    def bulk_add(
        self,
        documents: List[Dict[str, Any]],
        embeddings_key: str = "embedding",
        id_key: str = "id",
        batch_size: int = 1000,
        show_progress: bool = True
    ) -> List[str]:
        """
        Add multiple documents with embedded vectors in bulk.
        
        Args:
            documents: List of documents (each with an embedding field)
            embeddings_key: Key for embeddings in document
            id_key: Key for document ID
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of document IDs
        """
        all_ids = []
        
        # Process in batches
        iterator = range(0, len(documents), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Adding documents", unit="batch")
        
        for i in iterator:
            batch = documents[i:i+batch_size]
            
            # Extract vectors and IDs
            batch_vectors = []
            batch_docs = []
            batch_ids = []
            
            for doc in batch:
                if embeddings_key in doc:
                    # Get vector
                    vector = doc[embeddings_key]
                    
                    # Ensure correct format
                    if isinstance(vector, list):
                        vector = np.array(vector, dtype=np.float32)
                    
                    # Check dimension
                    if vector.shape[0] != self.dimension:
                        logger.warning(f"Skipping document with incorrect vector dimension: "
                                      f"{vector.shape[0]} (expected {self.dimension})")
                        continue
                    
                    # Get ID if present
                    doc_id = doc.get(id_key, f"doc_{len(self.doc_ids) + len(batch_ids)}")
                    
                    # Remove embedding from document to avoid duplication
                    doc_copy = doc.copy()
                    if embeddings_key in doc_copy:
                        del doc_copy[embeddings_key]
                    
                    batch_vectors.append(vector)
                    batch_docs.append(doc_copy)
                    batch_ids.append(doc_id)
            
            # Add batch
            if batch_vectors:
                vectors = np.array(batch_vectors, dtype=np.float32)
                added_ids = self.add(vectors, batch_docs, batch_ids)
                all_ids.extend(added_ids)
        
        return all_ids
    
    def rebuild_index(self) -> None:
        """
        Rebuild the index from stored documents.
        
        This is useful after many deletions or updates.
        """
        # Get all active documents
        active_docs = []
        active_ids = []
        active_vectors = []
        
        for doc_id in self.doc_ids:
            if doc_id in self.documents and not self.documents[doc_id].get("_deleted", False):
                # We need vectors for rebuilding
                vector_idx = self.id_to_index[doc_id]
                if vector_idx >= 0:
                    # Extract vector from the index - not easily possible with FAISS
                    # We would need the original vectors, which we don't store
                    # Just warn that rebuilding is not fully supported
                    logger.warning("Rebuilding index is not fully supported without original vectors.")
                    return
        
        # If we had the vectors, we would do:
        # 1. Create a new index
        # 2. Add all active vectors and documents
        # 3. Replace the current index
    
    def hybrid_search(
        self,
        query_vector: np.ndarray,
        query_terms: List[str],
        k: int = 10,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and text matching.
        
        This method combines dense retrieval (vector similarity) with
        sparse retrieval (term matching) for better results.
        
        Args:
            query_vector: Dense query vector
            query_terms: List of query terms for sparse matching
            k: Number of results to return
            alpha: Weight between dense (0) and sparse (1) scores
            
        Returns:
            List of search results
        """
        # This would typically use a specialized hybrid search index
        # or combine results from separate dense and sparse indices
        
        # For this implementation, we'll do a simple approach:
        # 1. Get more results than needed from vector search
        # 2. Calculate term matching scores
        # 3. Combine scores and re-rank
        
        # Get more results from vector search
        dense_results = self.search(query_vector, k=k*3)
        
        # Calculate term matching scores
        for result in dense_results:
            doc = result["document"]
            text = ""
            
            # Extract text from document (assuming common field names)
            if "text" in doc:
                text = doc["text"]
            elif "content" in doc:
                text = doc["content"]
            elif "body" in doc:
                text = doc["body"]
            elif "title" in doc:
                text = doc["title"]
                
            # Calculate sparse score using simple term matching
            # In a real implementation, this would use BM25 or similar
            sparse_score = 0.0
            if text:
                text_lower = text.lower()
                # Count matching terms
                matches = sum(1 for term in query_terms if term.lower() in text_lower)
                # Normalize by number of terms
                if query_terms:
                    sparse_score = matches / len(query_terms)
            
            # Combine scores
            dense_score = result["score"]
            combined_score = (1 - alpha) * dense_score + alpha * sparse_score
            
            # Update score
            result["dense_score"] = dense_score
            result["sparse_score"] = sparse_score
            result["score"] = combined_score
        
        # Re-rank based on combined score
        dense_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k
        return dense_results[:k]