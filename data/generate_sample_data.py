# data/generate_sample_data.py
import pandas as pd
import os

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("data/product_images", exist_ok=True)

# Create sample market data with 20 products
market_data = {
    'product_id': list(range(1, 21)),
    'product_name': [f"Product {i}" for i in range(1, 21)],
    'category': ['Electronics', 'Food', 'Beverage', 'Personal Care', 'Home Goods'] * 4,
    'price_point': ['Premium', 'Budget', 'Mid-range', 'Premium', 'Budget'] * 4,
    'consumer_sentiment': [0.92, 0.45, 0.78, 0.65, 0.82, 0.71, 0.39, 0.56, 0.88, 0.77, 
                          0.42, 0.69, 0.83, 0.61, 0.74, 0.49, 0.93, 0.85, 0.57, 0.68],
    'market_share': [15.2, 5.7, 8.3, 10.5, 7.2, 6.8, 4.3, 9.1, 12.7, 8.9, 
                     3.2, 7.5, 11.3, 6.1, 9.8, 4.7, 14.3, 10.9, 5.2, 8.6],
    'competitor_analysis': [
        "Main competitors include Samsung, Apple with premium features driving growth",
        "Price competition from store brands affecting market position negatively",
        "Stable market with loyal customer base despite new entrants",
        "Luxury positioning successful but facing pressure from mid-tier brands",
        "Value proposition strong but quality concerns affecting retention",
        "Innovative features driving growth despite premium pricing",
        "Struggling with price wars and low margins in saturated market",
        "Strong brand loyalty offsetting competitive pressures",
        "Premium positioning with innovative features maintaining lead",
        "Mid-market positioning successful with good quality/price ratio",
        "Losing market share to more innovative competitors",
        "Gaining traction with eco-friendly positioning despite higher price",
        "Premium brand seeing good growth in higher income segments",
        "Stable performance with consistent quality but limited innovation",
        "Strong growth through effective social media marketing",
        "Budget option struggling with quality perception issues",
        "Category leader with strong innovation pipeline",
        "Effective premium positioning with quality backing claims",
        "New entrant gaining share through aggressive pricing",
        "Well-established brand with loyal customer base"
    ],
    'trend_keywords': [
        "wireless, smart-home, connectivity",
        "organic, affordable, bulk",
        "sugar-free, natural, functional",
        "natural, chemical-free, sustainable",
        "sustainable, multi-purpose, space-saving",
        "AI-enabled, voice-control, premium",
        "discount, basics, value",
        "convenient, ready-to-eat, healthy",
        "premium, experience, quality",
        "family-size, cost-effective, practical",
        "budget, essential, simple",
        "eco-friendly, plant-based, sustainable",
        "luxury, status, exclusive",
        "reliable, traditional, trusted",
        "innovative, social-media-popular, trendy",
        "affordable, basic-function, accessible",
        "cutting-edge, early-adopter, high-performance",
        "quality, durability, worth-the-price",
        "competitive-price, adequate-quality, accessible",
        "heritage, consistent, familiar"
    ]
}

# Save as CSV
pd.DataFrame(market_data).to_csv('data/market_data.csv', index=False)
print("Sample market data created successfully!")