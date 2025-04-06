# serving/ray_serve.py
import os
import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple

import ray
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
import torch
from PIL import Image
import io
import base64
import traceback

# Import project modules
from models.fusion.fusion_model import MultimodalFusionModel
from models.visual.clip_model import CLIPWrapper
from models.text.roberta_model import RoBERTaWrapper
from models.optimization.onnx_export import ONNXExporter
from causal.structural_model import StructuralCausalModel
from causal.causal_features import CausalFeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/ray_serve.log")
    ]
)
logger = logging.getLogger("ray_serve")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configuration
DEFAULT_MODEL_PATHS = {
    "fusion_model": os.environ.get("FUSION_MODEL_PATH", "models/fusion_model.pt"),
    "causal_model": os.environ.get("CAUSAL_MODEL_PATH", "models/causal_model.pt"),
    "clip_model": os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32"),
    "roberta_model": os.environ.get("ROBERTA_MODEL", "roberta-base")
}

# Model cache directory
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Ray Serve deployment configuration
RAY_SERVE_CONFIG = {
    "address": os.environ.get("RAY_ADDRESS", "auto"),
    "http_host": os.environ.get("HTTP_HOST", "0.0.0.0"),
    "http_port": int(os.environ.get("HTTP_PORT", "8000")),
    "prometheus_metrics": True
}

# Model loading utilities
def load_fusion_model(model_path: str, device: Optional[str] = None) -> MultimodalFusionModel:
    """Load the multimodal fusion model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fusion model not found at {model_path}")
    
    logger.info(f"Loading fusion model from {model_path}")
    model = MultimodalFusionModel.load(model_path, device=device)
    return model

def load_causal_model(model_path: str) -> StructuralCausalModel:
    """Load the causal model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Causal model not found at {model_path}")
    
    logger.info(f"Loading causal model from {model_path}")
    return StructuralCausalModel.load(model_path)

def load_clip_model(model_name: str, device: Optional[str] = None) -> CLIPWrapper:
    """Load the CLIP model."""
    logger.info(f"Loading CLIP model {model_name}")
    return CLIPWrapper(model_name, device=device)

def load_roberta_model(model_name: str, device: Optional[str] = None) -> RoBERTaWrapper:
    """Load the RoBERTa model."""
    logger.info(f"Loading RoBERTa model {model_name}")
    return RoBERTaWrapper(model_name, device=device)

def get_device() -> str:
    """Get the device to use for model deployment."""
    device = os.environ.get("DEVICE", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def decode_image(image_data: str) -> Image.Image:
    """Decode base64-encoded image data."""
    # Remove data URL prefix if present
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    # Decode base64 data
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes))

# Ray Serve deployment class for the fusion model
@serve.deployment(
    ray_actor_options={"num_gpus": 0.25}, 
    max_concurrent_queries=10,
    autoscaling_config={"min_replicas": 1, "max_replicas": 3}
)
class FusionModelDeployment:
    def __init__(self, 
                 fusion_model_path: str = DEFAULT_MODEL_PATHS["fusion_model"],
                 clip_model_name: str = DEFAULT_MODEL_PATHS["clip_model"],
                 roberta_model_name: str = DEFAULT_MODEL_PATHS["roberta_model"]):
        # Get device
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self.fusion_model = load_fusion_model(fusion_model_path, device=self.device)
        self.clip_model = load_clip_model(clip_model_name, device=self.device)
        self.roberta_model = load_roberta_model(roberta_model_name, device=self.device)
        
        # Initialize statistics
        self.stats = {
            "total_requests": 0,
            "prediction_latency_ms": [],
            "last_request_time": 0,
            "model_load_time": time.time(),
        }
        
        logger.info("Fusion model deployment initialized successfully")
    
    async def predict_engagement(self, text_content: str, image_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict engagement from text and optional image.
        
        Args:
            text_content: Text content to analyze
            image_data: Base64-encoded image data (optional)
            
        Returns:
            Dictionary with engagement prediction results
        """
        start_time = time.time()
        
        # Process text
        text_features = self.roberta_model.encode_text(text_content)
        
        # Process image if provided
        if image_data:
            try:
                image = decode_image(image_data)
                visual_features = self.clip_model.encode_images(image)
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                # Use zero tensor for missing image
                visual_features = torch.zeros(
                    (1, self.clip_model.model.config.projection_dim),
                    device=self.clip_model.device
                )
        else:
            # Use zero tensor for missing image
            visual_features = torch.zeros(
                (1, self.clip_model.model.config.projection_dim),
                device=self.clip_model.device
            )
        
        # Predict engagement
        engagement = self.fusion_model.predict_engagement(
            visual_features, text_features
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats["total_requests"] += 1
        self.stats["prediction_latency_ms"].append(processing_time * 1000)
        self.stats["last_request_time"] = time.time()
        
        # Keep only last 100 latency measurements
        if len(self.stats["prediction_latency_ms"]) > 100:
            self.stats["prediction_latency_ms"] = self.stats["prediction_latency_ms"][-100:]
        
        # Create response
        response = {
            "engagement_prediction": engagement,
            "text_analyzed": True,
            "image_analyzed": image_data is not None,
            "processing_time_ms": processing_time * 1000
        }
        
        return response
    
    async def predict_sentiment(self, text_content: str, image_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict sentiment from text and optional image.
        
        Args:
            text_content: Text content to analyze
            image_data: Base64-encoded image data (optional)
            
        Returns:
            Dictionary with sentiment analysis results
        """
        start_time = time.time()
        
        # Process text
        text_features = self.roberta_model.encode_text(text_content)
        
        # Process image if provided
        if image_data:
            try:
                image = decode_image(image_data)
                visual_features = self.clip_model.encode_images(image)
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                # Use zero tensor for missing image
                visual_features = torch.zeros(
                    (1, self.clip_model.model.config.projection_dim),
                    device=self.clip_model.device
                )
        else:
            # Use zero tensor for missing image
            visual_features = torch.zeros(
                (1, self.clip_model.model.config.projection_dim),
                device=self.clip_model.device
            )
        
        # Predict sentiment
        sentiment = self.fusion_model.predict_sentiment(
            visual_features, text_features
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats["total_requests"] += 1
        self.stats["prediction_latency_ms"].append(processing_time * 1000)
        self.stats["last_request_time"] = time.time()
        
        # Create response
        response = {
            "sentiment_analysis": sentiment,
            "text_analyzed": True,
            "image_analyzed": image_data is not None,
            "processing_time_ms": processing_time * 1000
        }
        
        return response
    
    async def extract_content_features(self, text_content: str, image_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract multimodal content features.
        
        Args:
            text_content: Text content to analyze
            image_data: Base64-encoded image data (optional)
            
        Returns:
            Dictionary with extracted content features
        """
        start_time = time.time()
        
        # Process text
        text_features = self.roberta_model.encode_text(text_content)
        
        # Process image if provided
        if image_data:
            try:
                image = decode_image(image_data)
                visual_features = self.clip_model.encode_images(image)
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                # Use zero tensor for missing image
                visual_features = torch.zeros(
                    (1, self.clip_model.model.config.projection_dim),
                    device=self.clip_model.device
                )
        else:
            # Use zero tensor for missing image
            visual_features = torch.zeros(
                (1, self.clip_model.model.config.projection_dim),
                device=self.clip_model.device
            )
        
        # Extract content features
        content_features = self.fusion_model.extract_content_features(
            visual_features, text_features
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats["total_requests"] += 1
        self.stats["prediction_latency_ms"].append(processing_time * 1000)
        self.stats["last_request_time"] = time.time()
        
        # Create response
        response = {
            "content_features": content_features.tolist(),
            "text_analyzed": True,
            "image_analyzed": image_data is not None,
            "processing_time_ms": processing_time * 1000
        }
        
        return response
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        # Calculate statistics
        avg_latency = np.mean(self.stats["prediction_latency_ms"]) if self.stats["prediction_latency_ms"] else 0
        
        stats = {
            "total_requests": self.stats["total_requests"],
            "avg_latency_ms": avg_latency,
            "model_uptime_seconds": time.time() - self.stats["model_load_time"],
            "fusion_model": {
                "type": self.fusion_model.__class__.__name__,
                "device": self.device
            },
            "clip_model": {
                "name": self.clip_model.model_name,
                "device": self.clip_model.device
            },
            "roberta_model": {
                "name": self.roberta_model.model_name,
                "device": self.roberta_model.device
            }
        }
        
        return stats
    
    async def __call__(self, request: Request) -> Dict[str, Any]:
        """Handle incoming requests."""
        try:
            # Parse request body
            request_json = await request.json()
            endpoint = request_json.get("endpoint", "predict_engagement")
            
            # Route to appropriate endpoint
            if endpoint == "predict_engagement":
                return await self.predict_engagement(
                    request_json.get("text_content", ""),
                    request_json.get("image_data")
                )
            elif endpoint == "predict_sentiment":
                return await self.predict_sentiment(
                    request_json.get("text_content", ""),
                    request_json.get("image_data")
                )
            elif endpoint == "extract_features":
                return await self.extract_content_features(
                    request_json.get("text_content", ""),
                    request_json.get("image_data")
                )
            elif endpoint == "stats":
                return await self.get_stats()
            else:
                return {"error": f"Unknown endpoint: {endpoint}"}
        except Exception as e:
            logger.error(f"Error processing fusion model request: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "status": "error"}

# Ray Serve deployment class for the causal model
@serve.deployment(
    ray_actor_options={"num_cpus": 1}, 
    max_concurrent_queries=20,
    autoscaling_config={"min_replicas": 1, "max_replicas": 2}
)
class CausalModelDeployment:
    def __init__(self, causal_model_path: str = DEFAULT_MODEL_PATHS["causal_model"]):
        # Load causal model
        self.causal_model = load_causal_model(causal_model_path)
        
        # Initialize statistics
        self.stats = {
            "total_requests": 0,
            "analysis_latency_ms": [],
            "last_request_time": 0,
            "model_load_time": time.time(),
        }
        
        logger.info("Causal model deployment initialized successfully")
    
    async def analyze_causal_factors(
        self, 
        content_features: Dict[str, float], 
        target_engagement: Optional[float] = None,
        constraints: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze causal factors for audience engagement.
        
        Args:
            content_features: Dictionary of content features
            target_engagement: Target engagement level (optional)
            constraints: Feature constraints as {feature: [min, max]} (optional)
            
        Returns:
            Dictionary with causal analysis results
        """
        start_time = time.time()
        
        # Create DataFrame from features
        data = pd.DataFrame([content_features])
        
        # Get causal effects
        effects = self.causal_model.estimate_all_effects(
            data, "engagement", min_effect=0.05
        )
        
        # Convert to simplified format
        causal_factors = {}
        for feature, effect_info in effects.items():
            causal_factors[feature] = {
                "effect_size": effect_info["causal_effect"],
                "significance": effect_info.get("p_value", 1.0) < 0.05,
                "p_value": effect_info.get("p_value", None)
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
            selector = CausalFeatureSelector(self.causal_model)
            recommendations = selector.generate_feature_recommendations(
                target_engagement,
                content_features,
                constraints_dict
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats["total_requests"] += 1
        self.stats["analysis_latency_ms"].append(processing_time * 1000)
        self.stats["last_request_time"] = time.time()
        
        # Keep only last 100 latency measurements
        if len(self.stats["analysis_latency_ms"]) > 100:
            self.stats["analysis_latency_ms"] = self.stats["analysis_latency_ms"][-100:]
        
        # Create response
        response = {
            "causal_factors": causal_factors,
            "recommendations": recommendations,
            "processing_time_ms": processing_time * 1000
        }
        
        return response
    
    async def do_intervention(
        self,
        data: Dict[str, float],
        interventions: Dict[str, float],
        outcome_var: str = "engagement"
    ) -> Dict[str, Any]:
        """
        Perform a do-intervention: P(outcome | do(intervention)).
        
        Args:
            data: Dictionary of feature values
            interventions: Dictionary of intervention values
            outcome_var: Name of outcome variable
            
        Returns:
            Dictionary with intervention results
        """
        start_time = time.time()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Perform intervention
        outcome = self.causal_model.do_intervention(df, interventions, outcome_var)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats["total_requests"] += 1
        self.stats["analysis_latency_ms"].append(processing_time * 1000)
        self.stats["last_request_time"] = time.time()
        
        # Create response
        response = {
            "outcome": outcome,
            "interventions": interventions,
            "processing_time_ms": processing_time * 1000
        }
        
        return response
    
    async def counterfactual_analysis(
        self,
        content_features: Dict[str, float],
        interventions: Dict[str, float],
        outcome_var: str = "engagement"
    ) -> Dict[str, Any]:
        """
        Perform counterfactual analysis.
        
        Args:
            content_features: Dictionary of content features
            interventions: Dictionary of interventions to apply
            outcome_var: Name of outcome variable
            
        Returns:
            Dictionary with counterfactual analysis results
        """
        start_time = time.time()
        
        # Convert to DataFrame
        data = pd.DataFrame([content_features])
        
        # Create series from dictionary
        instance = pd.Series(content_features)
        
        # Perform counterfactual analysis
        result = self.causal_model.counterfactual_analysis(
            instance, interventions, outcome_var
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats["total_requests"] += 1
        self.stats["analysis_latency_ms"].append(processing_time * 1000)
        self.stats["last_request_time"] = time.time()
        
        # Create response
        response = {
            "counterfactual_analysis": result,
            "processing_time_ms": processing_time * 1000
        }
        
        return response
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        # Calculate statistics
        avg_latency = np.mean(self.stats["analysis_latency_ms"]) if self.stats["analysis_latency_ms"] else 0
        
        stats = {
            "total_requests": self.stats["total_requests"],
            "avg_latency_ms": avg_latency,
            "model_uptime_seconds": time.time() - self.stats["model_load_time"],
            "causal_model": {
                "type": self.causal_model.__class__.__name__,
                "discovery_method": self.causal_model.discovery_method,
                "num_features": len(self.causal_model.feature_names)
            }
        }
        
        return stats
    
    async def __call__(self, request: Request) -> Dict[str, Any]:
        """Handle incoming requests."""
        try:
            # Parse request body
            request_json = await request.json()
            endpoint = request_json.get("endpoint", "analyze_causal_factors")
            
            # Route to appropriate endpoint
            if endpoint == "analyze_causal_factors":
                return await self.analyze_causal_factors(
                    request_json.get("content_features", {}),
                    request_json.get("target_engagement"),
                    request_json.get("constraints")
                )
            elif endpoint == "do_intervention":
                return await self.do_intervention(
                    request_json.get("data", {}),
                    request_json.get("interventions", {}),
                    request_json.get("outcome_var", "engagement")
                )
            elif endpoint == "counterfactual_analysis":
                return await self.counterfactual_analysis(
                    request_json.get("content_features", {}),
                    request_json.get("interventions", {}),
                    request_json.get("outcome_var", "engagement")
                )
            elif endpoint == "stats":
                return await self.get_stats()
            else:
                return {"error": f"Unknown endpoint: {endpoint}"}
        except Exception as e:
            logger.error(f"Error processing causal model request: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "status": "error"}

# Main API handler
@serve.deployment(route_prefix="/api")
class APIRouter:
    def __init__(self, fusion_model_handler, causal_model_handler):
        self.fusion_model = fusion_model_handler
        self.causal_model = causal_model_handler
    
    async def __call__(self, request: Request) -> Response:
        # Parse path to determine endpoint
        path = request.url.path
        
        try:
            # Health check endpoint
            if path == "/api/health":
                return JSONResponse({
                    "status": "ok",
                    "timestamp": time.time()
                })
            
            # Parse request body for non-health endpoints
            try:
                request_json = await request.json()
            except Exception as e:
                return JSONResponse({
                    "error": f"Error parsing request body: {str(e)}",
                    "status": "error"
                }, status_code=400)
            
            # Route to appropriate model
            if path.startswith("/api/fusion"):
                # Route to fusion model endpoints
                request_json["endpoint"] = path.replace("/api/fusion/", "").rstrip("/")
                result = await self.fusion_model.remote(request)
                return JSONResponse(result)
            
            elif path.startswith("/api/causal"):
                # Route to causal model endpoints
                request_json["endpoint"] = path.replace("/api/causal/", "").rstrip("/")
                result = await self.causal_model.remote(request)
                return JSONResponse(result)
            
            elif path == "/api/stats":
                # Collect stats from all models
                fusion_stats = await self.fusion_model.remote({"endpoint": "stats"})
                causal_stats = await self.causal_model.remote({"endpoint": "stats"})
                
                return JSONResponse({
                    "fusion_model": fusion_stats,
                    "causal_model": causal_stats,
                    "timestamp": time.time()
                })
            
            else:
                return JSONResponse({
                    "error": f"Unknown endpoint: {path}",
                    "status": "error"
                }, status_code=404)
                
        except Exception as e:
            logger.error(f"Error routing request: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse({
                "error": f"Error processing request: {str(e)}",
                "status": "error"
            }, status_code=500)

# Create deployment graph
def create_deployment():
    # Create model deployments
    fusion_model = FusionModelDeployment.bind()
    causal_model = CausalModelDeployment.bind()
    
    # Create router
    router = APIRouter.bind(fusion_model, causal_model)
    
    return router

# Main function to start Ray Serve
def main():
    # Initialize Ray (if not already running)
    ray.init(address=RAY_SERVE_CONFIG["address"], ignore_reinit_error=True)
    
    # Deploy the application
    serve.start(
        http_host=RAY_SERVE_CONFIG["http_host"],
        http_port=RAY_SERVE_CONFIG["http_port"],
        detached=True
    )
    
    # Run the application
    handle = serve.run(create_deployment())
    
    logger.info(f"Ray Serve deployment running on http://{RAY_SERVE_CONFIG['http_host']}:{RAY_SERVE_CONFIG['http_port']}")
    
    # Keep the process running in non-detached mode
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Ray Serve deployment")
        serve.shutdown()

if __name__ == "__main__":
    main()