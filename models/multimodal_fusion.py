# models/multimodal_fusion.py
from typing import Dict, Any, List, Optional
from models.rag_pipeline import MarketIntelligenceRAG
from models.image_analyzer import ProductImageAnalyzer

class MultimodalMarketIntelligence:
    """Combines text and image analysis for market intelligence."""
    
    def __init__(self):
        """Initialize components."""
        self.rag = MarketIntelligenceRAG()
        self.image_analyzer = ProductImageAnalyzer()
    
    def analyze_product(
        self, 
        product_description: str, 
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a product using text description and optional image.
        
        Args:
            product_description: Text description
            image_path: Path to product image (optional)
            
        Returns:
            Dictionary with analysis results
        """
        # Text-based analysis
        text_analysis = self.rag.query(
            f"Analyze this product: {product_description}. "
            f"What are key market trends related to this product? "
            f"What is the competitive landscape?"
        )
        
        # Image analysis if image provided
        image_analysis = None
        if image_path:
            try:
                image_analysis = self.image_analyzer.analyze_image(image_path)
            except Exception as e:
                print(f"Warning: Image analysis failed: {e}")
                image_analysis = None
            
        # Combine results
        if image_analysis:
            # Use prompt engineering to combine insights
            combined_query = (
                f"Product description: {product_description}\n"
                f"Visual analysis: This product appears to be a {image_analysis['top_category']} "
                f"with attributes including {', '.join(image_analysis['top_attributes'])}.\n"
                f"Based on both the description and visual analysis, provide market insights, "
                f"competitive positioning, and trend alignment."
            )
            
            combined_analysis = self.rag.query(combined_query)
            
            result = {
                "text_analysis": text_analysis["answer"],
                "image_analysis": image_analysis,
                "combined_analysis": combined_analysis["answer"],
                "confidence": 0.87  # Placeholder for demo
            }
        else:
            result = {
                "text_analysis": text_analysis["answer"],
                "confidence": 0.82  # Placeholder for demo
            }
            
        return result
    
    def identify_market_trends(
        self, 
        category: Optional[str] = None,
        with_competitors: bool = False
    ) -> Dict[str, Any]:
        """
        Identify market trends for a category.
        
        Args:
            category: Product category (optional)
            with_competitors: Include competitor analysis
            
        Returns:
            Dictionary with trend analysis
        """
        trends = self.rag.analyze_market_trends(category)
        
        if with_competitors and category:
            competitor_query = f"Who are the main competitors in the {category} category and what are their strategies?"
            competitor_analysis = self.rag.query(competitor_query)
            
            result = {
                "trend_analysis": trends["trends"],
                "competitor_analysis": competitor_analysis["answer"],
                "confidence": trends.get("confidence", 0.85)
            }
        else:
            result = {
                "trend_analysis": trends["trends"],
                "confidence": trends.get("confidence", 0.85)
            }
            
        return result