# tests/test_rag.py
import unittest
import numpy as np
import torch
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from rag.retriever import DenseRetriever, SparseRetriever, HybridRetriever
from rag.vector_store import VectorStore
from rag.reranker import Reranker
from rag.generator import ContentGenerator


class TestVectorStore(unittest.TestCase):
    """Test the VectorStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a small vector store
        self.vector_store = VectorStore(
            dimension=4,
            index_type="flat",
            metric="cosine",
            storage_dir=self.temp_dir
        )
        
        # Create some test vectors and documents
        self.vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.documents = [
            {"text": "Document about topic A"},
            {"text": "Document about topic B"},
            {"text": "Document about topic C"},
            {"text": "Document about topic D"}
        ]
        
        # Add to vector store
        self.doc_ids = self.vector_store.add(self.vectors, self.documents)
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_add_and_search(self):
        """Test adding vectors and searching."""
        # Check that documents were added
        self.assertEqual(len(self.doc_ids), 4)
        
        # Search for vectors
        query = np.array([1.0, 0.1, 0.1, 0.1], dtype=np.float32)
        results = self.vector_store.search(query, k=2)
        
        # Should return 2 results
        self.assertEqual(len(results), 2)
        
        # First result should be closest to the query (doc 0)
        self.assertEqual(results[0]["id"], self.doc_ids[0])
        
        # Check that scores were calculated
        self.assertIn("score", results[0])
        self.assertGreater(results[0]["score"], 0)
    
    def test_batch_search(self):
        """Test batch searching."""
        queries = np.array([
            [1.0, 0.1, 0.1, 0.1],
            [0.1, 1.0, 0.1, 0.1]
        ], dtype=np.float32)
        
        results = self.vector_store.batch_search(queries, k=2)
        
        # Should return 2 batches of results
        self.assertEqual(len(results), 2)
        
        # Each batch should have 2 results
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(len(results[1]), 2)
        
        # First batch should have doc 0 as the top result
        self.assertEqual(results[0][0]["id"], self.doc_ids[0])
        
        # Second batch should have doc 1 as the top result
        self.assertEqual(results[1][0]["id"], self.doc_ids[1])
    
    def test_save_and_load(self):
        """Test saving and loading the vector store."""
        # Save the vector store
        save_path = self.vector_store.save()
        
        # Load the vector store
        loaded_store = VectorStore.load(self.temp_dir)
        
        # Check that the loaded store has the same data
        self.assertEqual(loaded_store.stats["num_vectors"], 4)
        
        # Search with loaded store
        query = np.array([1.0, 0.1, 0.1, 0.1], dtype=np.float32)
        results = loaded_store.search(query, k=1)
        
        # Should return the same top result
        self.assertEqual(results[0]["id"], self.doc_ids[0])


class TestDenseRetriever(unittest.TestCase):
    """Test the DenseRetriever class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the embedding model and tokenizer
        self.patcher1 = patch("transformers.AutoModel.from_pretrained")
        self.patcher2 = patch("transformers.AutoTokenizer.from_pretrained")
        
        mock_model = MagicMock()
        mock_model.config.hidden_size = 4
        
        # Mock the model output
        last_hidden_state = torch.zeros(1, 10, 4)
        last_hidden_state[:, 0, :] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        
        mock_output = MagicMock()
        mock_output.last_hidden_state = last_hidden_state
        mock_model.return_value = mock_output
        
        self.mock_model = self.patcher1.start()
        self.mock_model.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.ones((1, 10), dtype=torch.long),
            "attention_mask": torch.ones((1, 10), dtype=torch.long)
        })
        
        self.mock_tokenizer = self.patcher2.start()
        self.mock_tokenizer.return_value = mock_tokenizer
        
        # Create a vector store
        self.vector_store = VectorStore(dimension=4, index_type="flat", metric="cosine")
        
        # Create the retriever
        self.retriever = DenseRetriever(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            vector_store=self.vector_store
        )
        
        # Add test documents
        self.vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.documents = [
            {"text": "Document about topic A"},
            {"text": "Document about topic B"},
            {"text": "Document about topic C"},
            {"text": "Document about topic D"}
        ]
        
        self.vector_store.add(self.vectors, self.documents)
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher1.stop()
        self.patcher2.stop()
    
    def test_encode_query(self):
        """Test query encoding."""
        query_embedding = self.retriever.encode_query("Test query")
        
        # Should return a vector of the right shape
        self.assertEqual(query_embedding.shape, (1, 4))
    
    def test_search(self):
        """Test retrieval search."""
        # Mock the encode_query method to return a known vector
        self.retriever.encode_query = MagicMock(return_value=np.array([[1.0, 0.1, 0.1, 0.1]]))
        
        # Search for documents
        results = self.retriever.search("Test query", top_k=2)
        
        # Should return 2 results
        self.assertEqual(len(results), 2)
        
        # First result should be document 0
        self.assertEqual(results[0]["document"]["text"], "Document about topic A")
        
        # Should add the query to results
        self.assertEqual(results[0]["query"], "Test query")


class TestGenerator(unittest.TestCase):
    """Test the ContentGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the language model and tokenizer
        self.patcher1 = patch("transformers.AutoModelForCausalLM.from_pretrained")
        self.patcher2 = patch("transformers.AutoTokenizer.from_pretrained")
        
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=torch.ones((1, 20), dtype=torch.long))
        
        self.mock_model = self.patcher1.start()
        self.mock_model.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.ones((1, 10), dtype=torch.long),
            "attention_mask": torch.ones((1, 10), dtype=torch.long)
        })
        mock_tokenizer.decode = MagicMock(side_effect=lambda x, **kwargs: "Generated text" if len(x) > 10 else "Prompt")
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token_id = 2
        
        self.mock_tokenizer = self.patcher2.start()
        self.mock_tokenizer.return_value = mock_tokenizer
        
        # Create the generator
        self.generator = ContentGenerator(
            model_name="meta-llama/Llama-3-8B-Instruct",
            max_new_tokens=100
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher1.stop()
        self.patcher2.stop()
    
    def test_format_prompt(self):
        """Test prompt formatting."""
        query = "What are the best practices for content marketing?"
        contexts = [
            {"text": "Content should be relevant to the audience."},
            {"text": "Regular publishing schedule is important."}
        ]
        
        prompt = self.generator.format_prompt(query, contexts)
        
        # Should include the query
        self.assertIn(query, prompt)
        
        # Should include context information
        self.assertIn("Content should be relevant", prompt)
        self.assertIn("Regular publishing schedule", prompt)
    
    def test_generate(self):
        """Test content generation."""
        query = "What are the best practices for content marketing?"
        contexts = [
            {"text": "Content should be relevant to the audience."},
            {"text": "Regular publishing schedule is important."}
        ]
        
        # Generate content
        response = self.generator.generate(query, contexts)
        
        # Should return a string
        self.assertIsInstance(response, str)
        self.assertEqual(response, "Generated text")


class TestHybridRetriever(unittest.TestCase):
    """Test the HybridRetriever class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dense and sparse retrievers
        self.mock_dense = MagicMock(spec=DenseRetriever)
        self.mock_sparse = MagicMock(spec=SparseRetriever)
        self.mock_reranker = MagicMock(spec=Reranker)
        
        # Mock search results
        dense_results = [
            {"id": "doc1", "score": 0.9, "document": {"text": "Doc 1"}, "query": "test"},
            {"id": "doc2", "score": 0.7, "document": {"text": "Doc 2"}, "query": "test"}
        ]
        
        sparse_results = [
            {"id": "doc3", "score": 0.8, "document": {"text": "Doc 3"}, "query": "test"},
            {"id": "doc2", "score": 0.6, "document": {"text": "Doc 2"}, "query": "test"}
        ]
        
        self.mock_dense.search.return_value = dense_results
        self.mock_sparse.search.return_value = sparse_results
        
        # Create the hybrid retriever
        self.retriever = HybridRetriever(
            dense_retriever=self.mock_dense,
            sparse_retriever=self.mock_sparse,
            reranker=self.mock_reranker,
            fusion_method="interpolation",
            alpha=0.5
        )
    
    def test_search(self):
        """Test hybrid search."""
        # Mock reranker to return input results
        self.mock_reranker.rerank.return_value = [
            {"id": "doc1", "score": 0.9, "document": {"text": "Doc 1"}, "query": "test"},
            {"id": "doc3", "score": 0.8, "document": {"text": "Doc 3"}, "query": "test"}
        ]
        
        # Search for documents
        results = self.retriever.search("test query", top_k=2)
        
        # Should return 2 results
        self.assertEqual(len(results), 2)
        
        # Should call both retrievers
        self.mock_dense.search.assert_called_once()
        self.mock_sparse.search.assert_called_once()
        
        # Should call reranker
        self.mock_reranker.rerank.assert_called_once()
    
    def test_interpolation_fusion(self):
        """Test interpolation fusion method."""
        dense_results = [
            {"id": "doc1", "score": 0.9, "document": {"text": "Doc 1"}, "query": "test"},
            {"id": "doc2", "score": 0.7, "document": {"text": "Doc 2"}, "query": "test"}
        ]
        
        sparse_results = [
            {"id": "doc3", "score": 0.8, "document": {"text": "Doc 3"}, "query": "test"},
            {"id": "doc2", "score": 0.6, "document": {"text": "Doc 2"}, "query": "test"}
        ]
        
        results = self.retriever._interpolation_fusion(dense_results, sparse_results, alpha=0.5)
        
        # Should have 3 unique documents
        self.assertEqual(len(results), 3)
        
        # Doc2 should have combined scores
        for result in results:
            if result["id"] == "doc2":
                # Should have both dense and sparse scores
                self.assertIn("dense_score", result)
                self.assertIn("sparse_score", result)


if __name__ == '__main__':
    unittest.main()