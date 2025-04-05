"""Utility functions for the market intelligence system."""

def get_template_response(query: str, context: str) -> str:
    """Generate a response for a query based on context using templates."""
    if "trend" in query.lower():
        return _generate_trend_response(query, context)
    elif "competitor" in query.lower() or "competitive" in query.lower():
        return _generate_competitor_response(query, context)
    elif "product" in query.lower() or "analyze" in query.lower():
        return _generate_product_analysis(query, context)
    else:
        return _generate_general_response(query, context)

def _extract_keywords(text: str) -> list:
    """Extract key terms from text."""
    # Simple keyword extraction by frequency
    common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "as", "of"}
    words = text.lower().replace(",", " ").replace(".", " ").replace(":", " ").split()
    keywords = [word for word in words if word not in common_words and len(word) > 3]
    
    # Count frequency
    word_counts = {}
    for word in keywords:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency
    sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [k for k, v in sorted_keywords[:10]]  # Return top 10 keywords

def _generate_trend_response(query: str, context: str) -> str:
    """Generate a response about market trends."""
    # Extract keywords as trends
    keywords = _extract_keywords(context)
    
    # Look for trend keywords specifically in the context
    trend_lines = []
    for line in context.split("\n"):
        if "trend" in line.lower() or "keyword" in line.lower():
            trend_lines.append(line.strip())
    
    # Format response
    if "category" in query.lower():
        category = query.split("in")[1].split("products")[0].strip() if "in" in query else "this category"
        response = f"Based on market analysis, the key trends in {category} are:\n\n"
    else:
        response = "Based on market analysis, the key market trends are:\n\n"
    
    # Add extracted trends
    if trend_lines:
        for i, line in enumerate(trend_lines[:3], 1):
            response += f"{i}. {line}\n"
    else:
        for i, keyword in enumerate(keywords[:5], 1):
            response += f"{i}. {keyword.title()} - showing significant momentum in the market\n"
    
    response += "\nThese trends indicate evolving consumer preferences and market opportunities."
    return response

def _generate_competitor_response(query: str, context: str) -> str:
    """Generate a response about competitive landscape."""
    # Extract competitor information from context
    competitor_lines = []
    for line in context.split("\n"):
        if "competitor" in line.lower() or "competition" in line.lower():
            competitor_lines.append(line.strip())
    
    # Format response
    if "category" in query.lower():
        category = query.split("in the")[1].split("category")[0].strip() if "in the" in query else "this category"
        response = f"The competitive landscape in the {category} category shows:\n\n"
    elif "product" in query.lower():
        product = query.split("for")[1].strip().rstrip("?")
        response = f"The competitive landscape for {product} reveals:\n\n"
    else:
        response = "The overall market competitive landscape shows:\n\n"
    
    # Add extracted competitor information
    if competitor_lines:
        for i, line in enumerate(competitor_lines[:3], 1):
            response += f"{i}. {line}\n"
    else:
        response += "1. Market leaders maintaining strong positions through innovation\n"
        response += "2. Emerging competitors disrupting with unique value propositions\n"
        response += "3. Price competition intensifying in certain segments\n"
    
    response += "\nThis competitive environment suggests opportunities for differentiation through quality, innovation, or targeted positioning."
    return response

def _generate_product_analysis(query: str, context: str) -> str:
    """Generate a product analysis response."""
    # Extract product information
    product_info = {}
    category = None
    price_point = None
    
    for line in context.split("\n"):
        line = line.strip()
        if line.startswith("Product:"):
            product_info["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Category:"):
            category = line.split(":", 1)[1].strip()
            product_info["category"] = category
        elif line.startswith("Price Point:"):
            price_point = line.split(":", 1)[1].strip()
            product_info["price_point"] = price_point
        elif line.startswith("Consumer Sentiment:"):
            sentiment = line.split(":", 1)[1].strip()
            product_info["sentiment"] = sentiment
        elif line.startswith("Market Share:"):
            share = line.split(":", 1)[1].strip()
            product_info["market_share"] = share
    
    # Format response
    if "name" in product_info:
        response = f"Analysis for {product_info['name']}:\n\n"
    else:
        response = "Product Analysis:\n\n"
    
    response += f"Category: {category or 'Unknown'}\n"
    response += f"Positioning: {price_point or 'Mid-range'}\n\n"
    
    # Add market insights
    response += "Market Insights:\n"
    response += f"1. This product {_get_sentiment_phrase(product_info.get('sentiment', '0.5'))} consumer sentiment\n"
    
    if "market_share" in product_info:
        response += f"2. Current market share is approximately {product_info['market_share']}%\n"
    
    # Add recommendations
    response += "\nRecommendations:\n"
    if price_point and price_point.lower() == "premium":
        response += "1. Emphasize quality and exclusive features to justify premium pricing\n"
        response += "2. Target high-value customer segments with tailored messaging\n"
    elif price_point and price_point.lower() == "budget":
        response += "1. Highlight value proposition and cost efficiency\n"
        response += "2. Optimize operations to maintain competitive pricing\n"
    else:
        response += "1. Consider feature differentiation to stand out in the market\n"
        response += "2. Evaluate pricing strategy against competitive landscape\n"
    
    return response

def _generate_general_response(query: str, context: str) -> str:
    """Generate a general response based on context."""
    # Extract key information from context
    keywords = _extract_keywords(context)
    
    # Format response
    response = f"Based on market intelligence analysis:\n\n"
    
    # Add general insights from context
    response += "Key Insights:\n"
    for i, keyword in enumerate(keywords[:3], 1):
        response += f"{i}. {keyword.title()} is an important factor in this market\n"
    
    response += "\nThis analysis provides a foundation for strategic decision-making in product development, marketing, and competitive positioning."
    return response

def _get_sentiment_phrase(sentiment_value: str) -> str:
    """Convert sentiment value to descriptive phrase."""
    try:
        sentiment = float(sentiment_value)
        if sentiment >= 0.8:
            return "enjoys extremely positive"
        elif sentiment >= 0.7:
            return "has strong positive"
        elif sentiment >= 0.6:
            return "shows above average"
        elif sentiment >= 0.5:
            return "has moderate"
        elif sentiment >= 0.4:
            return "has mixed"
        else:
            return "faces challenges with"
    except ValueError:
        return "has varying"
