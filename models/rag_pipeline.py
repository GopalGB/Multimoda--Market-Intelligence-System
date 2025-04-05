import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from models.utils import get_template_response

class MarketIntelligenceRAG:
    """Simplified RAG system using sentence embeddings."""
    
    def __init__(self, data_path: str = "data/market_data.csv"):
        """Initialize the RAG pipeline with market data."""
        self.data_path = data_path
        
        # Load model for embeddings
        print("Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load and process market data
        self._load_data()
        
        # Create embeddings for all documents
        self._create_embeddings()
        
    def _load_data(self):
        """Load and process market data from CSV."""
        df = pd.read_csv(self.data_path)
        
        # Convert each row to a document
        self.documents = []
        for _, row in df.iterrows():
            content = f"""
            Product: {row['product_name']}
            Category: {row['category']}
            Price Point: {row['price_point']}
            Consumer Sentiment: {row['consumer_sentiment']}
            Market Share: {row['market_share']}
            Competitor Analysis: {row['competitor_analysis']}
            Trend Keywords: {row['trend_keywords']}
            """
            self.documents.append({
                'content': content,
                'metadata': dict(row)
            })
    
    def _create_embeddings(self):
        """Create embeddings for all documents."""
        texts = [doc['content'] for doc in self.documents]
        self.embeddings = self.model.encode(texts)
    
    def _get_relevant_documents(self, query: str, k: int = 3):
        """Retrieve relevant documents for a query."""
        # Create query embedding
        query_embedding = self.model.encode(query)
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k documents
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.documents[i] for i in top_indices]
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """Process a market intelligence query."""
        # Get relevant documents
        relevant_docs = self._get_relevant_documents(query_text)
        
        # Extract relevant information
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        
        # Generate a response using templates
        response = get_template_response(query_text, context)
        
        # Format the response
        result = {
            "answer": response,
            "source_documents": [doc['content'] for doc in relevant_docs]
        }
        
        return result

    def analyze_market_trends(self, category: str = None) -> Dict[str, Any]:
        """Analyze market trends, optionally filtered by category."""
        query = f"What are the emerging trends in {category} products?" if category else "What are the main emerging trends across all product categories?"
        
        # Filter by category if provided
        if category:
            relevant_docs = [doc for doc in self.documents if doc['metadata']['category'] == category]
            if not relevant_docs:  # Fallback if no matches
                relevant_docs = self._get_relevant_documents(query)
        else:
            relevant_docs = self._get_relevant_documents(query)
        
        # Extract trend keywords from relevant documents
        trends = []
        for doc in relevant_docs:
            if 'trend_keywords' in doc['metadata']:
                trends.extend(doc['metadata']['trend_keywords'].split(', '))
        
        # Generate a response
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        response = get_template_response(query, context)
        
        # Format the response
        result = {
            "trends": response,
            "confidence": 0.87,  # Placeholder for demo
            "source_documents": [doc['content'] for doc in relevant_docs]
        }
        
        return result
        
    def analyze_competitor_landscape(self, product_name: str = None, category: str = None) -> Dict[str, Any]:
        """Analyze the competitive landscape for a product or category."""
        if product_name:
            query = f"What is the competitive landscape for {product_name}?"
            # Try to find exact product first
            relevant_docs = [doc for doc in self.documents if doc['metadata']['product_name'] == product_name]
            if not relevant_docs:  # Fallback if no exact match
                relevant_docs = self._get_relevant_documents(query)
        elif category:
            query = f"What is the competitive landscape in the {category} category?"
            # Filter by category
            relevant_docs = [doc for doc in self.documents if doc['metadata']['category'] == category]
            if not relevant_docs:  # Fallback if no matches
                relevant_docs = self._get_relevant_documents(query)
        else:
            query = "What are the main competitive dynamics across the market?"
            relevant_docs = self._get_relevant_documents(query, k=5)
        
        # Extract competitive information
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        response = get_template_response(query, context)
        
        # Format the response
        result = {
            "competitive_analysis": response,
            "confidence": 0.85,  # Placeholder for demo
            "source_documents": [doc['content'] for doc in relevant_docs]
        }
        
        return result
