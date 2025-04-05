# tests/test_multimodal.py
from models.multimodal_fusion import MultimodalMarketIntelligence
import os

def main():
    print("Initializing multimodal system...")
    # Initialize the multimodal system
    mmi = MultimodalMarketIntelligence()

    # Test product analysis with text only
    print("\n=== PRODUCT ANALYSIS (TEXT ONLY) ===")
    text_result = mmi.analyze_product(
        "A premium wireless headphone with noise cancellation and 24-hour battery life"
    )
    print(text_result["text_analysis"])
    print(f"Confidence: {text_result['confidence']}")

    # Test market trend identification
    print("\n=== MARKET TREND ANALYSIS ===")
    trends = mmi.identify_market_trends(category="Electronics", with_competitors=True)
    print("Trend Analysis:")
    print(trends["trend_analysis"])
    print("\nCompetitor Analysis:")
    if "competitor_analysis" in trends:
        print(trends["competitor_analysis"])
    else:
        print("No competitor analysis available")

if __name__ == "__main__":
    main()