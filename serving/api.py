# serving/api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Query, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Union, Optional, Any, Tuple
import torch
import numpy as np
from PIL import Image
import io
import json
import time
import os
import logging
import asyncio
from datetime import datetime, timedelta
import uuid
import hashlib
import shutil
from pathlib import Path
import aiohttp
import aiofiles
import requests
from functools import lru_cache
import traceback

# Project imports
from models.fusion.fusion_model import MultimodalFusionModel
from models.visual.clip_model import CLIPWrapper
from models.visual.visual_features import VisualFeatureExtractor
from models.text.roberta_model import RoBERTaWrapper
from models.text.text_features import TextFeatureExtractor
from rag.retriever import HybridRetriever
from rag.generator import ContentGenerator
from causal.structural_model import StructuralCausalModel
from causal.counterfactual import CounterfactualAnalyzer
from causal.causal_features import CausalFeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api_logs.log")
    ]
)
logger = logging.getLogger("audience_intelligence_api")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Create FastAPI application
app = FastAPI(
    title="Cross-Modal Audience Intelligence Platform API",
    description="API for analyzing audience engagement using multimodal AI",
    version="1.0.0",
    docs_url=None  # Disable default docs to use custom docs with authentication
)

# API Versioning
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
class RateLimiter:
    def __init__(self, rate_limit: int = 100, period: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            rate_limit: Maximum requests per period
            period: Time period in seconds
        """
        self.rate_limit = rate_limit
        self.period = period
        self.clients = {}
    
    async def __call__(self, request: Request):
        client_ip = request.client.host
        
        # Get current timestamp
        now = time.time()
        
        # Initialize client if not seen before
        if client_ip not in self.clients:
            self.clients[client_ip] = {
                "count": 0,
                "start_time": now
            }
        
        # Reset count if period has elapsed
        if now - self.clients[client_ip]["start_time"] > self.period:
            self.clients[client_ip] = {
                "count": 0,
                "start_time": now
            }
        
        # Increment count
        self.clients[client_ip]["count"] += 1
        
        # Check if rate limit exceeded
        if self.clients[client_ip]["count"] > self.rate_limit:
            time_to_wait = self.period - (now - self.clients[client_ip]["start_time"])
            headers = {"Retry-After": str(int(time_to_wait))}
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {int(time_to_wait)} seconds.",
                headers=headers
            )

# Add rate limiter middleware
rate_limiter = RateLimiter()
app.middleware("http")(rate_limiter)

# Global variables for models
model_store = {
    "fusion_model": None,
    "clip_model": None,
    "roberta_model": None,
    "visual_extractor": None,
    "text_extractor": None,
    "retriever": None,
    "content_generator": None,
    "causal_model": None,
    "model_versions": {}
}

# Directory for saving uploaded content
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Model configurations
MODEL_CONFIG = {
    "visual_model": os.environ.get("VISUAL_MODEL", "openai/clip-vit-base-patch32"),
    "text_model": os.environ.get("TEXT_MODEL", "roberta-base"),
    "fusion_model_path": os.environ.get("FUSION_MODEL_PATH", "models/fusion_model.pt"),
    "causal_model_path": os.environ.get("CAUSAL_MODEL_PATH", "models/causal_model.pt"),
    "model_cache_dir": os.environ.get("MODEL_CACHE_DIR", "model_cache"),
    "model_download_urls": {
        "fusion_model": os.environ.get("FUSION_MODEL_URL", ""),
        "causal_model": os.environ.get("CAUSAL_MODEL_URL", "")
    }
}

# Cache directory for models
os.makedirs(MODEL_CONFIG["model_cache_dir"], exist_ok=True)

# API key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# API keys (in production, use a database or environment variables)
API_KEYS = {
    os.environ.get("API_KEY", "test_key"): {
        "client": "test",
        "tier": "standard",
        "rate_limit": 100
    }
}

# Tiers and their limits
TIER_LIMITS = {
    "standard": {
        "batch_size_limit": 50,
        "max_content_length": 5000,
        "max_image_size": 5 * 1024 * 1024,  # 5MB
        "allowed_endpoints": ["engagement", "sentiment", "batch"]
    },
    "premium": {
        "batch_size_limit": 200,
        "max_content_length": 20000,
        "max_image_size": 20 * 1024 * 1024,  # 20MB
        "allowed_endpoints": ["engagement", "sentiment", "batch", "causal", "counterfactual", "insights"]
    },
    "enterprise": {
        "batch_size_limit": 1000,
        "max_content_length": 50000,
        "max_image_size": 50 * 1024 * 1024,  # 50MB
        "allowed_endpoints": ["engagement", "sentiment", "batch", "causal", "counterfactual", "insights"]
    }
}

# Pydantic models for API
class ApiResponse(BaseModel):
    """Base API response model."""
    success: bool = True
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = API_VERSION

class ErrorResponse(ApiResponse):
    """Error response model."""
    success: bool = False
    error_code: str
    detail: str

class EngagementRequest(BaseModel):
    """Request model for engagement prediction."""
    text_content: str = Field(..., description="Text content to analyze")
    visual_content_url: Optional[str] = Field(None, description="URL to visual content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    
    @validator('text_content')
    def validate_text_length(cls, value, values, **kwargs):
        """Validate text length."""
        if len(value) > 50000:  # Max length handled by the API
            raise ValueError("Text content too long")
        return value

class EngagementResponse(ApiResponse):
    """Response model for engagement prediction."""
    engagement_prediction: Dict[str, Any]
    text_analyzed: bool
    image_analyzed: bool
    processing_time_ms: float
    
class ContentFeatures(BaseModel):
    """Content features for causal analysis."""
    features: Dict[str, float] = Field(..., description="Content features")
    target_engagement: Optional[float] = Field(None, description="Target engagement level")
    constraints: Optional[Dict[str, List[float]]] = Field(None, description="Feature constraints")
    model_version: Optional[str] = Field(None, description="Specific model version to use")

class CausalResponse(ApiResponse):
    """Response model for causal analysis."""
    causal_factors: Dict[str, Dict[str, Any]]
    recommendations: Optional[Dict[str, Dict[str, Any]]] = None
    processing_time_ms: float

class CounterfactualRequest(BaseModel):
    """Request model for counterfactual analysis."""
    content_features: Dict[str, float] = Field(..., description="Current content features")
    interventions: Dict[str, float] = Field(..., description="Feature interventions")
    outcome_variable: str = Field("engagement", description="Outcome variable name")
    model_version: Optional[str] = Field(None, description="Specific model version to use")

class CounterfactualResponse(ApiResponse):
    """Response model for counterfactual analysis."""
    counterfactual_result: Dict[str, Any]
    processing_time_ms: float

class InsightRequest(BaseModel):
    """Request model for content insights."""
    content_text: str = Field(..., description="Content text")
    content_type: str = Field(..., description="Content type (e.g., video, article)")
    target_audience: Optional[str] = Field(None, description="Target audience")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    
    @validator('content_text')
    def validate_text_length(cls, value, values, **kwargs):
        """Validate text length."""
        if len(value) > 50000:  # Max length handled by the API
            raise ValueError("Content text too long")
        return value

class InsightResponse(ApiResponse):
    """Response model for content insights."""
    insights: str
    content_type: str
    target_audience: Optional[str] = None
    processing_time_ms: float

class SentimentResponse(ApiResponse):
    """Response model for sentiment analysis."""
    sentiment_analysis: Dict[str, Any]
    text_analyzed: bool
    image_analyzed: bool
    processing_time_ms: float

class BatchItem(BaseModel):
    """Model for a batch item."""
    id: Optional[str] = Field(None, description="Unique ID for the item")
    text: str = Field(..., description="Text content to analyze")
    image_url: Optional[str] = Field(None, description="URL to image content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""
    content_items: List[BatchItem] = Field(..., description="List of content items to analyze")
    analysis_type: str = Field("engagement", description="Type of analysis to perform")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    callback_url: Optional[str] = Field(None, description="URL to call with results when complete")
    
    @validator('content_items')
    def validate_batch_size(cls, value, values, **kwargs):
        """Validate batch size."""
        if len(value) > 1000:  # Max batch size
            raise ValueError("Batch size too large")
        return value
    
    @validator('analysis_type')
    def validate_analysis_type(cls, value, values, **kwargs):
        """Validate analysis type."""
        valid_types = ["engagement", "sentiment", "causal", "counterfactual", "insights"]
        if value not in valid_types:
            raise ValueError(f"Invalid analysis type. Must be one of: {', '.join(valid_types)}")
        return value

class BatchSubmitResponse(ApiResponse):
    """Response model for batch submission."""
    batch_id: str
    num_items: int
    analysis_type: str
    status: str
    estimated_completion_time: Optional[str] = None

class BatchProgress(BaseModel):
    """Model for batch progress."""
    completed: int
    total: int
    percent: float

class BatchStatusResponse(ApiResponse):
    """Response model for batch status."""
    batch_id: str
    status: str
    progress: Optional[BatchProgress] = None
    errors: Optional[List[Dict[str, str]]] = None
    processing_time_seconds: Optional[float] = None
    num_results: Optional[int] = None
    results_url: Optional[str] = None

# Authentication dependency
async def get_api_key(api_key_header: str = Header(None, alias=API_KEY_NAME)):
    """
    Verify API key and get client info.
    
    Args:
        api_key_header: API key from request header
        
    Returns:
        Dictionary with client info
        
    Raises:
        HTTPException: If API key is invalid
    """
    if api_key_header is None:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": API_KEY_NAME}
        )
    
    if api_key_header not in API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": API_KEY_NAME}
        )
    
    return {
        "client": API_KEYS[api_key_header]["client"],
        "tier": API_KEYS[api_key_header]["tier"],
        "rate_limit": API_KEYS[api_key_header]["rate_limit"]
    }

# Authorization dependency for specific endpoints
def check_endpoint_access(endpoint: str):
    """
    Create dependency to check endpoint access based on client tier.
    
    Args:
        endpoint: Endpoint name to check access for
        
    Returns:
        Dependency function
    """
    async def _check_access(client_info: Dict[str, Any] = Depends(get_api_key)):
        tier = client_info["tier"]
        if endpoint not in TIER_LIMITS[tier]["allowed_endpoints"]:
            raise HTTPException(
                status_code=403,
                detail=f"Access to {endpoint} not allowed for {tier} tier",
                headers={"WWW-Authenticate": API_KEY_NAME}
            )
        return client_info
    
    return _check_access

# Download model if needed
async def download_model(model_name: str, url: str, cache_dir: str) -> str:
    """
    Download model from URL if not in cache.
    
    Args:
        model_name: Name of the model
        url: URL to download the model from
        cache_dir: Directory to cache the model
        
    Returns:
        Path to the downloaded model
    """
    if not url:
        logger.warning(f"No download URL provided for {model_name}")
        return None
    
    # Create model directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get cached path
    url_hash = hashlib.md5(url.encode()).hexdigest()
    model_path = os.path.join(cache_dir, f"{model_name}_{url_hash}.pt")
    
    # Check if already downloaded
    if os.path.exists(model_path):
        logger.info(f"Model {model_name} already in cache")
        return model_path
    
    # Download if not in cache
    logger.info(f"Downloading {model_name} from {url}")
    
    try:
        # Create temporary file
        temp_path = model_path + ".tmp"
        
        # Download
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download {model_name}: {response.status}")
                    return None
                
                content_length = response.content_length
                downloaded = 0
                
                # Stream to file
                async with aiofiles.open(temp_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        logger.info(f"Downloaded {downloaded}/{content_length} bytes")
        
        # Rename to final path
        shutil.move(temp_path, model_path)
        
        logger.info(f"Model {model_name} downloaded successfully")
        return model_path
    
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {str(e)}")
        return None

# Load models
async def load_models():
    """
    Load all required models asynchronously.
    """
    # Log start time
    start_time = time.time()
    logger.info("Loading models...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Download models if needed
        fusion_model_path = MODEL_CONFIG["fusion_model_path"]
        if not os.path.exists(fusion_model_path) and MODEL_CONFIG["model_download_urls"]["fusion_model"]:
            fusion_model_path = await download_model(
                "fusion_model",
                MODEL_CONFIG["model_download_urls"]["fusion_model"],
                MODEL_CONFIG["model_cache_dir"]
            )
        
        causal_model_path = MODEL_CONFIG["causal_model_path"]
        if not os.path.exists(causal_model_path) and MODEL_CONFIG["model_download_urls"]["causal_model"]:
            causal_model_path = await download_model(
                "causal_model",
                MODEL_CONFIG["model_download_urls"]["causal_model"],
                MODEL_CONFIG["model_cache_dir"]
            )
        
        # Load visual models
        logger.info("Loading visual models...")
        model_store["clip_model"] = CLIPWrapper(MODEL_CONFIG["visual_model"], device=device)
        model_store["visual_extractor"] = VisualFeatureExtractor(MODEL_CONFIG["visual_model"], device=device)
        
        # Load text models
        logger.info("Loading text models...")
        model_store["roberta_model"] = RoBERTaWrapper(MODEL_CONFIG["text_model"], device=device)
        model_store["text_extractor"] = TextFeatureExtractor(MODEL_CONFIG["text_model"], device=device)
        
        # Load fusion model if available
        if fusion_model_path and os.path.exists(fusion_model_path):
            logger.info("Loading fusion model...")
            model_store["fusion_model"] = MultimodalFusionModel.load(fusion_model_path, device=device)
            # Get model version info
            model_store["model_versions"]["fusion_model"] = {
                "path": fusion_model_path,
                "timestamp": datetime.fromtimestamp(os.path.getmtime(fusion_model_path)).isoformat(),
                "size": os.path.getsize(fusion_model_path)
            }
        else:
            logger.warning(f"Fusion model not found at {fusion_model_path}. Initializing new model.")
            model_store["fusion_model"] = MultimodalFusionModel(device=device)
            model_store["model_versions"]["fusion_model"] = {
                "path": "initialized_default",
                "timestamp": datetime.now().isoformat(),
                "size": 0
            }
        
        # Load RAG components
        logger.info("Initializing retriever and content generator...")
        model_store["retriever"] = HybridRetriever()
        model_store["content_generator"] = ContentGenerator()
        
        # Load causal model if available
        if causal_model_path and os.path.exists(causal_model_path):
            logger.info("Loading causal model...")
            with open(causal_model_path, 'rb') as f:
                model_store["causal_model"] = torch.load(f)
            # Get model version info
            model_store["model_versions"]["causal_model"] = {
                "path": causal_model_path,
                "timestamp": datetime.fromtimestamp(os.path.getmtime(causal_model_path)).isoformat(),
                "size": os.path.getsize(causal_model_path)
            }
        else:
            logger.warning(f"Causal model not found at {causal_model_path}.")
            model_store["causal_model"] = None
            model_store["model_versions"]["causal_model"] = {
                "path": "not_available",
                "timestamp": datetime.now().isoformat(),
                "size": 0
            }
        
        # Log completion
        load_time = time.time() - start_time
        logger.info(f"All models loaded successfully in {load_time:.2f} seconds")
        
        # Return models
        return model_store
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to load models: {str(e)}")

# Dependency to ensure models are loaded
async def get_models():
    """
    Get loaded models, loading them if needed.
    
    Returns:
        Dictionary with loaded models
    """
    if model_store["fusion_model"] is None:
        await load_models()
    return model_store

# Custom Swagger UI with authentication
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):
    """
    Custom Swagger UI page with API key authentication.
    """
    return get_swagger_ui_html(
        openapi_url=f"{request.url.path.replace('docs', '')}openapi.json",
        title=app.title + " - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.15.5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.15.5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running."""
    models_loaded = model_store["fusion_model"] is not None
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_loaded,
        "version": API_VERSION
    }

# Model info endpoint
@app.get(f"{API_PREFIX}/models/info")
async def model_info(
    client_info: Dict[str, Any] = Depends(get_api_key),
    models=Depends(get_models)
):
    """Get information about loaded models."""
    return {
        "models": model_store["model_versions"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timestamp": datetime.now().isoformat(),
        "version": API_VERSION
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting up API server")
    try:
        await load_models()
    except Exception as e:
        logger.error(f"Failed to load models on startup: {str(e)}")
        logger.error(traceback.format_exc())

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down API server")
    # Clean up temporary files
    try:
        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith("temp_"):
                filepath = os.path.join(UPLOAD_DIR, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}")

# Engagement prediction endpoint
@app.post(
    f"{API_PREFIX}/predict/engagement",
    response_model=EngagementResponse,
    summary="Predict audience engagement",
    description="Analyze content and predict audience engagement levels"
)
async def predict_engagement(
    request: Optional[EngagementRequest] = None,
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    models=Depends(get_models),
    client_info: Dict[str, Any] = Depends(check_endpoint_access("engagement"))
):
    """
    Predict engagement for content with text and optional image.
    
    Can accept either JSON body or form data with file upload.
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(f"Received engagement prediction request from {client_info['client']}")
        
        # Enforce tier limits
        tier = client_info["tier"]
        tier_limits = TIER_LIMITS[tier]
        
        # Handle form data
        if text is not None:
            text_content = text
            visual_content = image
            model_version = None
        # Handle JSON request
        elif request is not None:
            text_content = request.text_content
            visual_content = None
            model_version = request.model_version
            
            # Enforce content length limit
            if len(text_content) > tier_limits["max_content_length"]:
                raise HTTPException(
                    status_code=413,
                    detail=f"Text content exceeds maximum length for {tier} tier"
                )
            
            # Download image from URL if provided
            if request.visual_content_url:
                try:
                    response = requests.get(request.visual_content_url)
                    if response.status_code == 200:
                        visual_content = response.content
                        
                        # Check image size
                        if len(visual_content) > tier_limits["max_image_size"]:
                            raise HTTPException(
                                status_code=413,
                                detail=f"Image size exceeds maximum for {tier} tier"
                            )
                    else:
                        visual_content = None
                        logger.warning(f"Failed to download image from URL: {response.status_code}")
                except Exception as e:
                    visual_content = None
                    logger.warning(f"Failed to download image from URL: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="No content provided")
        
        # Process text content
        text_features = models["roberta_model"].encode_text(text_content)
        
        # Process visual content if available
        if visual_content:
            # Convert to PIL Image
            if isinstance(visual_content, UploadFile):
                # Check image size for uploaded file
                contents = await visual_content.read()
                if len(contents) > tier_limits["max_image_size"]:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image size exceeds maximum for {tier} tier"
                    )
                image = Image.open(io.BytesIO(contents))
            else:
                image = Image.open(io.BytesIO(visual_content))
            
            # Extract visual features
            visual_features = models["clip_model"].encode_images(image)
        else:
            # Use zero tensor for missing image
            visual_features = torch.zeros(
                (1, models["clip_model"].model.config.projection_dim),
                device=models["clip_model"].device
            )
        
        # Predict engagement using fusion model
        engagement = models["fusion_model"].predict_engagement(
            visual_features, text_features
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Format response
        response = EngagementResponse(
            engagement_prediction=engagement,
            text_analyzed=True,
            image_analyzed=visual_content is not None,
            processing_time_ms=processing_time * 1000,
            message="Engagement prediction successful"
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error predicting engagement: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to predict engagement: {str(e)}"
        )

# Causal analysis endpoint
@app.post(
    f"{API_PREFIX}/analyze/causal",
    response_model=CausalResponse,
    summary="Analyze causal factors",
    description="Identify causal factors affecting audience engagement"
)
async def analyze_causal_factors(
    content_features: ContentFeatures,
    models=Depends(get_models),
    client_info: Dict[str, Any] = Depends(check_endpoint_access("causal"))
):
    """
    Analyze causal factors affecting engagement.
    
    Identifies which content features have causal effects on engagement.
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(f"Received causal analysis request from {client_info['client']}")
        
        # Check if causal model is available
        if models["causal_model"] is None:
            raise HTTPException(status_code=503, detail="Causal model not available")
        
        # Create DataFrame from features
        import pandas as pd
        data = pd.DataFrame([content_features.features])
        
        # Get causal effects
        causal_model = models["causal_model"]
        effects = causal_model.estimate_all_effects(
            data, "engagement", min_effect=0.05
        )
        
        # Convert to simplified format
        causal_factors = {}
        for feature, effect_info in effects.items():
            causal_factors[feature] = {
                "effect_size": effect_info["causal_effect"],
                "significance": effect_info["p_value"] < 0.05,
                "p_value": effect_info["p_value"]
            }
        
        # Get feature recommendations if target provided
        recommendations = None
        if content_features.target_engagement is not None:
            # Convert constraints format
            constraints = None
            if content_features.constraints:
                constraints = {
                    feature: (min_val, max_val) 
                    for feature, (min_val, max_val) 
                    in content_features.constraints.items()
                }
            
            # Get recommendations
            selector = CausalFeatureSelector(causal_model)
            recommendations = selector.generate_feature_recommendations(
                content_features.target_engagement,
                content_features.features,
                constraints
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Format response
        response = CausalResponse(
            causal_factors=causal_factors,
            recommendations=recommendations,
            processing_time_ms=processing_time * 1000,
            message="Causal analysis successful"
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error analyzing causal factors: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze causal factors: {str(e)}"
        )

# Counterfactual analysis endpoint
@app.post(
    f"{API_PREFIX}/analyze/counterfactual",
    response_model=CounterfactualResponse,
    summary="Analyze counterfactual scenarios",
    description="Predict how changes to content features would affect engagement"
)
async def analyze_counterfactual(
    request: CounterfactualRequest,
    models=Depends(get_models),
    client_info: Dict[str, Any] = Depends(check_endpoint_access("counterfactual"))
):
    """
    Perform counterfactual analysis.
    
    Predicts how engagement would change if content features were different.
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(f"Received counterfactual analysis request from {client_info['client']}")
        
        # Check if causal model is available
        if models["causal_model"] is None:
            raise HTTPException(status_code=503, detail="Causal model not available")
        
        # Create DataFrame from features
        import pandas as pd
        data = pd.DataFrame([request.content_features])
        
        # Perform counterfactual analysis
        analyzer = CounterfactualAnalyzer(models["causal_model"])
        
        result = analyzer.generate_counterfactual(
            data, 
            request.interventions,
            request.outcome_variable,
            reference_values=request.content_features
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Format response
        response = CounterfactualResponse(
            counterfactual_result=result,
            processing_time_ms=processing_time * 1000,
            message="Counterfactual analysis successful"
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error analyzing counterfactual: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze counterfactual: {str(e)}"
        )

# Content insights endpoint
@app.post(
    f"{API_PREFIX}/insights",
    response_model=InsightResponse,
    summary="Generate content insights",
    description="Generate Nielsen audience insights for content"
)
async def generate_content_insights(
    request: InsightRequest,
    models=Depends(get_models),
    client_info: Dict[str, Any] = Depends(check_endpoint_access("insights"))
):
    """
    Generate insights for content.
    
    Uses RAG system to provide content-specific insights.
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(f"Received content insights request from {client_info['client']}")
        
        # Enforce tier limits
        tier = client_info["tier"]
        tier_limits = TIER_LIMITS[tier]
        
        # Enforce content length limit
        if len(request.content_text) > tier_limits["max_content_length"]:
            raise HTTPException(
                status_code=413,
                detail=f"Content text exceeds maximum length for {tier} tier"
            )
        
        # Check if retriever and generator are available
        if models["retriever"] is None or models["content_generator"] is None:
            raise HTTPException(status_code=503, detail="RAG system not available")
        
        # Format query based on content type and target audience
        query = f"Content analysis for {request.content_type}: {request.content_text}"
        if request.target_audience:
            query += f" targeting {request.target_audience} audience"
        
        # Retrieve relevant contexts
        contexts = models["retriever"].search(query, top_k=5)
        
        # Generate insights
        insights = models["content_generator"].generate(
            query=query,
            contexts=contexts,
            generation_kwargs={
                "temperature": 0.3,
                "max_tokens": 500
            }
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Format response
        response = InsightResponse(
            insights=insights,
            content_type=request.content_type,
            target_audience=request.target_audience,
            processing_time_ms=processing_time * 1000,
            message="Content insights generated successfully"
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating content insights: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate insights: {str(e)}"
        )

# Sentiment analysis endpoint
@app.post(
    f"{API_PREFIX}/analyze/sentiment",
    response_model=SentimentResponse,
    summary="Analyze content sentiment",
    description="Detect sentiment expressed in content"
)
async def analyze_sentiment(
    request: Optional[EngagementRequest] = None,
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    models=Depends(get_models),
    client_info: Dict[str, Any] = Depends(check_endpoint_access("sentiment"))
):
    """
    Analyze sentiment for content.
    
    Detects the sentiment expressed in the content.
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(f"Received sentiment analysis request from {client_info['client']}")
        
        # Enforce tier limits
        tier = client_info["tier"]
        tier_limits = TIER_LIMITS[tier]
        
        # Handle form data
        if text is not None:
            text_content = text
            visual_content = image
        # Handle JSON request
        elif request is not None:
            text_content = request.text_content
            visual_content = None
            
            # Enforce content length limit
            if len(text_content) > tier_limits["max_content_length"]:
                raise HTTPException(
                    status_code=413,
                    detail=f"Text content exceeds maximum length for {tier} tier"
                )
            
            # Download image from URL if provided
            if request.visual_content_url:
                try:
                    response = requests.get(request.visual_content_url)
                    if response.status_code == 200:
                        visual_content = response.content
                        
                        # Check image size
                        if len(visual_content) > tier_limits["max_image_size"]:
                            raise HTTPException(
                                status_code=413,
                                detail=f"Image size exceeds maximum for {tier} tier"
                            )
                    else:
                        visual_content = None
                        logger.warning(f"Failed to download image from URL: {response.status_code}")
                except Exception as e:
                    visual_content = None
                    logger.warning(f"Failed to download image from URL: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="No content provided")
        
        # Process text content
        text_features = models["roberta_model"].encode_text(text_content)
        
        # Process visual content if available
        if visual_content:
            # Convert to PIL Image
            if isinstance(visual_content, UploadFile):
                # Check image size for uploaded file
                contents = await visual_content.read()
                if len(contents) > tier_limits["max_image_size"]:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image size exceeds maximum for {tier} tier"
                    )
                image = Image.open(io.BytesIO(contents))
            else:
                image = Image.open(io.BytesIO(visual_content))
            
            # Extract visual features
            visual_features = models["clip_model"].encode_images(image)
        else:
            # Use zero tensor for missing image
            visual_features = torch.zeros(
                (1, models["clip_model"].model.config.projection_dim),
                device=models["clip_model"].device
            )
        
        # Predict sentiment using fusion model
        sentiment = models["fusion_model"].predict_sentiment(
            visual_features, text_features
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Format response
        response = SentimentResponse(
            sentiment_analysis=sentiment,
            text_analyzed=True,
            image_analyzed=visual_content is not None,
            processing_time_ms=processing_time * 1000,
            message="Sentiment analysis successful"
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze sentiment: {str(e)}"
        )

# Batch analysis endpoint
@app.post(
    f"{API_PREFIX}/analyze/batch",
    response_model=BatchSubmitResponse,
    summary="Submit batch analysis request",
    description="Analyze multiple content items asynchronously"
)
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    models=Depends(get_models),
    client_info: Dict[str, Any] = Depends(check_endpoint_access("batch"))
):
    """
    Analyze a batch of content items asynchronously.
    
    Performs the specified analysis type on multiple content items.
    """
    try:
        # Log request
        logger.info(f"Received batch analysis request from {client_info['client']} for {len(request.content_items)} items")
        
        # Enforce tier limits
        tier = client_info["tier"]
        tier_limits = TIER_LIMITS[tier]
        
        # Check batch size limit
        if len(request.content_items) > tier_limits["batch_size_limit"]:
            raise HTTPException(
                status_code=413,
                detail=f"Batch size exceeds maximum for {tier} tier"
            )
        
        # Check if analysis type is allowed for tier
        if request.analysis_type not in tier_limits["allowed_endpoints"]:
            raise HTTPException(
                status_code=403,
                detail=f"Analysis type {request.analysis_type} not allowed for {tier} tier"
            )
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Create batch directory
        batch_dir = os.path.join(UPLOAD_DIR, batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        
        # Save request for processing
        with open(os.path.join(batch_dir, "request.json"), "w") as f:
            json.dump(request.dict(), f)
        
        # Create initial status file
        with open(os.path.join(batch_dir, "status.json"), "w") as f:
            status = {
                "batch_id": batch_id,
                "status": "queued",
                "progress": {
                    "completed": 0,
                    "total": len(request.content_items),
                    "percent": 0
                },
                "timestamp": datetime.now().isoformat()
            }
            json.dump(status, f)
        
        # Start background processing
        background_tasks.add_task(
            process_batch,
            batch_id,
            request.content_items,
            request.analysis_type,
            request.callback_url,
            models,
            client_info
        )
        
        # Estimate completion time
        # Rough estimate: 0.5 seconds per item for engagement/sentiment, 
        # 2 seconds for causal/counterfactual
        per_item_time = 0.5  # Default for engagement/sentiment
        if request.analysis_type in ["causal", "counterfactual", "insights"]:
            per_item_time = 2.0
            
        estimated_seconds = len(request.content_items) * per_item_time
        estimated_completion = (datetime.now() + timedelta(seconds=estimated_seconds)).isoformat()
        
        # Return batch ID for status checking
        return BatchSubmitResponse(
            batch_id=batch_id,
            num_items=len(request.content_items),
            analysis_type=request.analysis_type,
            status="queued",
            estimated_completion_time=estimated_completion,
            message="Batch analysis submitted successfully"
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error starting batch analysis: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start batch analysis: {str(e)}"
        )

# Batch status endpoint
@app.get(
    f"{API_PREFIX}/analyze/batch/{{batch_id}}",
    response_model=BatchStatusResponse,
    summary="Get batch processing status",
    description="Check the status of a batch analysis request"
)
async def get_batch_status(
    batch_id: str,
    client_info: Dict[str, Any] = Depends(get_api_key)
):
    """
    Get the status of a batch analysis.
    
    Returns the processing status and results if completed.
    """
    try:
        # Log request
        logger.info(f"Received batch status request from {client_info['client']} for {batch_id}")
        
        # Check if batch directory exists
        batch_dir = os.path.join(UPLOAD_DIR, batch_id)
        if not os.path.exists(batch_dir):
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
        
        # Check for status file
        status_file = os.path.join(batch_dir, "status.json")
        if not os.path.exists(status_file):
            return BatchStatusResponse(
                batch_id=batch_id,
                status="unknown",
                message="Batch status not available"
            )
        
        # Read status file
        with open(status_file, "r") as f:
            status = json.load(f)
        
        # Add results URL if completed
        if status.get("status") == "completed":
            results_url = f"/api/{API_VERSION}/analyze/batch/{batch_id}/results"
            status["results_url"] = results_url
        
        # Convert to response model
        response = BatchStatusResponse(
            batch_id=status["batch_id"],
            status=status["status"],
            message=f"Batch status: {status['status']}",
            **{k: v for k, v in status.items() if k not in ["batch_id", "status"]}
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error checking batch status: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check batch status: {str(e)}"
        )

# Batch results endpoint
@app.get(
    f"{API_PREFIX}/analyze/batch/{{batch_id}}/results",
    summary="Get batch results",
    description="Get the results of a completed batch analysis"
)
async def get_batch_results(
    batch_id: str,
    client_info: Dict[str, Any] = Depends(get_api_key)
):
    """
    Get the results of a batch analysis.
    
    Returns the results if the batch has completed processing.
    """
    try:
        # Log request
        logger.info(f"Received batch results request from {client_info['client']} for {batch_id}")
        
        # Check if batch directory exists
        batch_dir = os.path.join(UPLOAD_DIR, batch_id)
        if not os.path.exists(batch_dir):
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
        
        # Check status
        status_file = os.path.join(batch_dir, "status.json")
        if not os.path.exists(status_file):
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} status not found")
        
        with open(status_file, "r") as f:
            status = json.load(f)
        
        if status.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Batch {batch_id} not yet completed (status: {status.get('status')})"
            )
        
        # Check for results file
        results_file = os.path.join(batch_dir, "results.json")
        if not os.path.exists(results_file):
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} results not found")
        
        # Read results file
        with open(results_file, "r") as f:
            results = json.load(f)
        
        # Return results
        return {
            "batch_id": batch_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "num_results": len(results)
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving batch results: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve batch results: {str(e)}"
        )

# Exception handler for validation errors
@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, exc: Exception):
    """
    Handle validation errors.
    """
    logger.error(f"Error processing request: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            message="An error occurred while processing your request",
            error_code="internal_error",
            detail=str(exc)
        ).dict()
    )

# Background processing function
async def process_batch(
    batch_id: str,
    content_items: List[BatchItem],
    analysis_type: str,
    callback_url: Optional[str],
    models: Dict[str, Any],
    client_info: Dict[str, Any]
):
    """
    Process a batch of content items in the background.
    
    Args:
        batch_id: ID of the batch
        content_items: List of content items to analyze
        analysis_type: Type of analysis to perform
        callback_url: URL to call when processing completes
        models: Dictionary of loaded models
        client_info: Client information
    """
    batch_dir = os.path.join(UPLOAD_DIR, batch_id)
    results_file = os.path.join(batch_dir, "results.json")
    status_file = os.path.join(batch_dir, "status.json")
    
    # Initialize results
    results = []
    errors = []
    
    # Update status to processing
    with open(status_file, "w") as f:
        status = {
            "batch_id": batch_id,
            "status": "processing",
            "progress": {
                "completed": 0,
                "total": len(content_items),
                "percent": 0
            },
            "timestamp": datetime.now().isoformat()
        }
        json.dump(status, f)
    
    # Process each item
    start_time = time.time()
    completed = 0
    
    try:
        for i, item in enumerate(content_items):
            try:
                # Get item ID
                item_id = item.id or str(i)
                
                # Process based on analysis type
                if analysis_type == "engagement":
                    result = await process_engagement_item(item, models)
                elif analysis_type == "sentiment":
                    result = await process_sentiment_item(item, models)
                elif analysis_type == "causal":
                    result = await process_causal_item(item, models)
                elif analysis_type == "counterfactual":
                    result = await process_counterfactual_item(item, models)
                elif analysis_type == "insights":
                    result = await process_insights_item(item, models)
                else:
                    raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
                # Add item ID to result
                result["item_id"] = item_id
                
                # Add to results
                results.append(result)
            
            except Exception as e:
                errors.append({
                    "item_id": item.id or str(i),
                    "error": str(e)
                })
                logger.error(f"Error processing batch item {i}: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Update completion count
            completed += 1
            
            # Update status periodically
            if i % 10 == 0 or i == len(content_items) - 1:
                with open(status_file, "w") as f:
                    json.dump({
                        "batch_id": batch_id,
                        "status": "processing",
                        "progress": {
                            "completed": completed,
                            "total": len(content_items),
                            "percent": round(completed / len(content_items) * 100, 1)
                        },
                        "timestamp": datetime.now().isoformat()
                    }, f)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save results
        with open(results_file, "w") as f:
            json.dump(results, f)
        
        # Update status to completed
        with open(status_file, "w") as f:
            json.dump({
                "batch_id": batch_id,
                "status": "completed",
                "progress": {
                    "completed": completed,
                    "total": len(content_items),
                    "percent": 100
                },
                "errors": errors,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "num_results": len(results)
            }, f)
        
        logger.info(f"Batch {batch_id} processing completed in {processing_time:.2f} seconds")
        
        # Call callback URL if provided
        if callback_url:
            try:
                async with aiohttp.ClientSession() as session:
                    callback_data = {
                        "batch_id": batch_id,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                        "num_results": len(results),
                        "errors": len(errors)
                    }
                    
                    async with session.post(callback_url, json=callback_data) as response:
                        if response.status != 200:
                            logger.warning(f"Callback to {callback_url} failed: {response.status}")
                        else:
                            logger.info(f"Callback to {callback_url} successful")
            except Exception as e:
                logger.error(f"Error calling callback URL {callback_url}: {str(e)}")
    
    except Exception as e:
        # Update status to failed
        with open(status_file, "w") as f:
            json.dump({
                "batch_id": batch_id,
                "status": "failed",
                "error": str(e),
                "progress": {
                    "completed": completed,
                    "total": len(content_items),
                    "percent": round(completed / len(content_items) * 100, 1) if len(content_items) > 0 else 0
                },
                "timestamp": datetime.now().isoformat()
            }, f)
        
        logger.error(f"Batch {batch_id} processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Call callback URL if provided
        if callback_url:
            try:
                async with aiohttp.ClientSession() as session:
                    callback_data = {
                        "batch_id": batch_id,
                        "status": "failed",
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e)
                    }
                    
                    async with session.post(callback_url, json=callback_data) as response:
                        if response.status != 200:
                            logger.warning(f"Callback to {callback_url} failed: {response.status}")
                        else:
                            logger.info(f"Callback to {callback_url} successful")
            except Exception as callback_error:
                logger.error(f"Error calling callback URL {callback_url}: {str(callback_error)}")

# Helper functions for batch processing
async def process_engagement_item(
    item: BatchItem,
    models: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single engagement analysis item."""
    # Process text content
    text_features = models["roberta_model"].encode_text(item.text)
    
    # Process visual content if available
    visual_content = None
    if item.image_url:
        try:
            response = requests.get(item.image_url)
            if response.status_code == 200:
                visual_content = response.content
        except Exception as e:
            logger.warning(f"Failed to download image from URL: {str(e)}")
    
    if visual_content:
        image = Image.open(io.BytesIO(visual_content))
        visual_features = models["clip_model"].encode_images(image)
    else:
        visual_features = torch.zeros(
            (1, models["clip_model"].model.config.projection_dim),
            device=models["clip_model"].device
        )
    
    # Predict engagement
    engagement = models["fusion_model"].predict_engagement(
        visual_features, text_features
    )
    
    # Return result
    return {
        "engagement_prediction": engagement,
        "text_analyzed": True,
        "image_analyzed": visual_content is not None,
        "metadata": item.metadata
    }

async def process_sentiment_item(
    item: BatchItem,
    models: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single sentiment analysis item."""
    # Process text content
    text_features = models["roberta_model"].encode_text(item.text)
    
    # Process visual content if available
    visual_content = None
    if item.image_url:
        try:
            response = requests.get(item.image_url)
            if response.status_code == 200:
                visual_content = response.content
        except Exception as e:
            logger.warning(f"Failed to download image from URL: {str(e)}")
    
    if visual_content:
        image = Image.open(io.BytesIO(visual_content))
        visual_features = models["clip_model"].encode_images(image)
    else:
        visual_features = torch.zeros(
            (1, models["clip_model"].model.config.projection_dim),
            device=models["clip_model"].device
        )
    
    # Predict sentiment
    sentiment = models["fusion_model"].predict_sentiment(
        visual_features, text_features
    )
    
    # Return result
    return {
        "sentiment_analysis": sentiment,
        "text_analyzed": True,
        "image_analyzed": visual_content is not None,
        "metadata": item.metadata
    }

async def process_causal_item(
    item: BatchItem,
    models: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single causal analysis item."""
    # Check if causal model is available
    if models["causal_model"] is None:
        raise ValueError("Causal model not available")
    
    # Extract features from metadata
    if not item.metadata or "features" not in item.metadata:
        raise ValueError("Features required in metadata for causal analysis")
    
    features = item.metadata["features"]
    target_engagement = item.metadata.get("target_engagement")
    constraints = item.metadata.get("constraints")
    
    # Create DataFrame from features
    import pandas as pd
    data = pd.DataFrame([features])
    
    # Get causal effects
    causal_model = models["causal_model"]
    effects = causal_model.estimate_all_effects(
        data, "engagement", min_effect=0.05
    )
    
    # Convert to simplified format
    causal_factors = {}
    for feature, effect_info in effects.items():
        causal_factors[feature] = {
            "effect_size": effect_info["causal_effect"],
            "significance": effect_info["p_value"] < 0.05,
            "p_value": effect_info["p_value"]
        }
    
    # Get feature recommendations if target provided
    recommendations = None
    if target_engagement is not None:
        # Convert constraints format
        constraints_dict = None
        if constraints:
            constraints_dict = {
                feature: (min_val, max_val) 
                for feature, (min_val, max_val) 
                in constraints.items()
            }
        
        # Get recommendations
        selector = CausalFeatureSelector(causal_model)
        recommendations = selector.generate_feature_recommendations(
            target_engagement,
            features,
            constraints_dict
        )
    
    # Return result
    return {
        "causal_factors": causal_factors,
        "recommendations": recommendations,
        "metadata": item.metadata
    }

async def process_counterfactual_item(
    item: BatchItem,
    models: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single counterfactual analysis item."""
    # Check if causal model is available
    if models["causal_model"] is None:
        raise ValueError("Causal model not available")
    
    # Extract features and interventions from metadata
    if not item.metadata or "content_features" not in item.metadata or "interventions" not in item.metadata:
        raise ValueError("Content features and interventions required in metadata for counterfactual analysis")
    
    content_features = item.metadata["content_features"]
    interventions = item.metadata["interventions"]
    outcome_variable = item.metadata.get("outcome_variable", "engagement")
    
    # Create DataFrame from features
    import pandas as pd
    data = pd.DataFrame([content_features])
    
    # Perform counterfactual analysis
    analyzer = CounterfactualAnalyzer(models["causal_model"])
    
    result = analyzer.generate_counterfactual(
        data, 
        interventions,
        outcome_variable,
        reference_values=content_features
    )
    
    # Return result
    return {
        "counterfactual_result": result,
        "metadata": item.metadata
    }

async def process_insights_item(
    item: BatchItem,
    models: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single insights analysis item."""
    # Check if retriever and generator are available
    if models["retriever"] is None or models["content_generator"] is None:
        raise ValueError("RAG system not available")
    
    # Extract content type and target audience from metadata
    content_type = item.metadata.get("content_type", "content")
    target_audience = item.metadata.get("target_audience")
    
    # Format query based on content type and target audience
    query = f"Content analysis for {content_type}: {item.text}"
    if target_audience:
        query += f" targeting {target_audience} audience"
    
    # Retrieve relevant contexts
    contexts = models["retriever"].search(query, top_k=5)
    
    # Generate insights
    insights = models["content_generator"].generate(
        query=query,
        contexts=contexts,
        generation_kwargs={
            "temperature": 0.3,
            "max_tokens": 500
        }
    )
    
    # Return result
    return {
        "insights": insights,
        "content_type": content_type,
        "target_audience": target_audience,
        "metadata": item.metadata
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)