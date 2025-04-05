from models.rag_pipeline import MarketIntelligenceRAG

# Initialize the RAG pipeline
print("Initializing RAG pipeline...")
rag = MarketIntelligenceRAG()

# Test a simple query
print("\nTesting simple query...")
result = rag.query("What are the trends in premium electronics?")
print(result["answer"])

# Test market trend analysis
print("\nTesting market trend analysis...")
trends = rag.analyze_market_trends(category="Electronics")
print(trends["trends"])