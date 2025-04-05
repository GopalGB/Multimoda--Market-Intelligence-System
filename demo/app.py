import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import json
import os

# Configure the page
st.set_page_config(
    page_title="Nielsen Market Intelligence System",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Nielsen Market Intelligence System")
st.markdown("""
This system analyzes consumer behavior across multiple data modalities (text, images) 
to identify market trends with high accuracy. Connect with Nielsen's proprietary data 
for actionable competitive intelligence.
""")

# API URL (change if deployed elsewhere)
API_URL = "http://localhost:8000"

# Check if API is running
def is_api_running():
    try:
        response = requests.get(f"{API_URL}")
        return response.status_code == 200
    except:
        return False

# Display warning if API isn't running
if not is_api_running():
    st.warning("❗ API server is not running. Please start the API server using: `python -m api.main`")
    st.stop()

# Sidebar for navigation
analysis_type = st.sidebar.radio(
    "Select Analysis Type:",
    ["Market Trend Analysis", "Product Analysis"]
)

# Function to call API
def call_api(endpoint, params=None, files=None):
    try:
        if files:
            response = requests.post(f"{API_URL}{endpoint}", data=params, files=files)
        else:
            response = requests.post(f"{API_URL}{endpoint}", params=params)
            
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

# Market Trend Analysis
if analysis_type == "Market Trend Analysis":
    st.header("Market Trend Analysis")
    
    # Input parameters
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox(
            "Select Product Category:",
            ["All Categories", "Electronics", "Food", "Beverage", "Personal Care", "Home Goods"]
        )
    with col2:
        include_competitors = st.checkbox("Include Competitor Analysis", value=True)
    
    # Process category for API
    api_category = None if category == "All Categories" else category
    
    # Execute analysis
    if st.button("Analyze Market Trends"):
        with st.spinner("Analyzing market trends..."):
            # Call API
            result = call_api(
                "/api/analyze_trends",
                params={
                    "category": api_category,
                    "with_competitors": include_competitors
                }
            )
            
            if result:
                # Display results
                st.subheader("Trend Analysis")
                st.write(result["trend_analysis"])
                
                # Display confidence
                st.metric("Analysis Confidence", f"{result['confidence']*100:.1f}%")
                
                # Create a simple visualization
                st.subheader("Market Trend Visualization")
                
                # Create a mock visualization for demo purposes
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Extract keywords from the trend analysis
                trend_text = result["trend_analysis"].lower()
                trend_keywords = [
                    "wireless", "smart", "premium", "affordable", "sustainable", 
                    "eco-friendly", "innovative", "traditional", "quality", "budget"
                ]
                
                # Count keyword occurrences as a simple metric
                keyword_counts = {}
                for keyword in trend_keywords:
                    count = trend_text.count(keyword)
                    if count > 0:
                        keyword_counts[keyword] = count
                
                # Sort by frequency
                sorted_keywords = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:6])
                
                # Create bar chart
                sns.barplot(x=list(sorted_keywords.keys()), y=list(sorted_keywords.values()), ax=ax)
                ax.set_title(f"Top Trend Keywords in {category if category != 'All Categories' else 'All Categories'}")
                ax.set_ylabel("Frequency")
                ax.set_xlabel("Trend Keywords")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Display competitor analysis if requested
                if include_competitors and "competitor_analysis" in result:
                    st.subheader("Competitor Analysis")
                    st.write(result["competitor_analysis"])

# Product Analysis
elif analysis_type == "Product Analysis":
    st.header("Product Analysis")
    
    # Input form
    with st.form("product_analysis_form"):
        description = st.text_area(
            "Product Description:",
            placeholder="Enter a detailed description of the product...",
            height=100
        )
        
        uploaded_file = st.file_uploader("Upload Product Image (optional):", type=["jpg", "jpeg", "png"])
        
        submit_button = st.form_submit_button("Analyze Product")
    
    # Execute analysis when form is submitted
    if submit_button:
        if not description:
            st.error("Please enter a product description")
        else:
            with st.spinner("Analyzing product..."):
                # Prepare files for API call
                files = None
                if uploaded_file is not None:
                    files = {"image": uploaded_file}
                
                # Call API
                result = call_api(
                    "/api/analyze_product",
                    params={"description": description},
                    files=files
                )
                
                if result:
                    # Display image if uploaded
                    if uploaded_file is not None:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(uploaded_file, caption="Uploaded Product Image", use_column_width=True)
                            
                            # Display image analysis if available
                            if "image_analysis" in result:
                                st.subheader("Visual Analysis")
                                
                                # Category analysis
                                st.write("**Category Classification:**")
                                st.write(f"Top category: **{result['image_analysis']['top_category']}**")
                                
                                # Create pie chart for categories
                                fig, ax = plt.subplots(figsize=(8, 8))
                                categories = result['image_analysis']['category_analysis']
                                plt.pie(
                                    list(categories.values()), 
                                    labels=list(categories.keys()), 
                                    autopct='%1.1f%%',
                                    startangle=90
                                )
                                plt.axis('equal')
                                plt.title("Product Category Analysis")
                                st.pyplot(fig)
                                
                                # Attribute analysis
                                st.write("**Top Attributes:**")
                                st.write(", ".join(result['image_analysis']['top_attributes']))
                        
                        with col2:
                            # Combined analysis
                            if "combined_analysis" in result:
                                st.subheader("Multimodal Analysis")
                                st.write(result["combined_analysis"])
                            else:
                                st.subheader("Text Analysis")
                                st.write(result["text_analysis"])
                    else:
                        # Text-only analysis
                        st.subheader("Analysis Results")
                        st.write(result["text_analysis"])
                    
                    # Display confidence
                    st.metric("Analysis Confidence", f"{result['confidence']*100:.1f}%")

# Run the Streamlit app
if __name__ == "__main__":
    # This is handled by Streamlit
    pass
