import streamlit as st
import requests
from typing import Dict, Any, List

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
to identify market trends with high accuracy.
""")

# API URL
API_URL = "http://127.0.0.1:8000"

# Check if API is running
def is_api_running():
    try:
        response = requests.get(f"{API_URL}")
        return response.status_code == 200
    except:
        return False

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
            
        response.raise_for_status()
        return response.json()
    except Exception as e:
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
                st.write(result.get("trend_analysis", result.get("trends", "")))
                
                # Display confidence
                confidence = result.get("confidence", 0)
                st.progress(confidence)
                st.write(f"Confidence: {confidence*100:.1f}%")
                
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
                    # Display results
                    if uploaded_file is not None and "image_analysis" in result:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(uploaded_file, caption="Uploaded Product Image", use_column_width=True)
                            
                            # Category analysis
                            st.subheader("Visual Analysis")
                            st.write(f"**Top category:** {result['image_analysis']['top_category']}")
                            st.write(f"**Top attributes:** {', '.join(result['image_analysis']['top_attributes'])}")
                            
                        with col2:
                            if "combined_analysis" in result:
                                st.subheader("Multimodal Analysis")
                                st.write(result["combined_analysis"])
                            else:
                                st.subheader("Text Analysis")
                                st.write(result["text_analysis"])
                    else:
                        st.subheader("Analysis Results")
                        st.write(result.get("text_analysis", ""))
                    
                    # Display confidence
                    confidence = result.get("confidence", 0)
                    st.progress(confidence)
                    st.write(f"Confidence: {confidence*100:.1f}%")
