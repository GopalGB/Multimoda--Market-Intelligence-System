"""
API routes for the Audience Intelligence platform.
Handles content analysis, causal inference, and optimization suggestions.
"""
import os
import io
import logging
import base64
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from pydantic import BaseModel, HttpUrl, validator
from typing import Optional, List, Dict, Any, Union

# Import model handlers
from models.model_manager import ModelManager

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize model manager (lazy loading)
model_manager = ModelManager()

# Validation models
class ContentAnalysisRequest(BaseModel):
    url: str
    title: str
    text: str
    primary_image: Optional[str] = None
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or len(v.strip()) < 50:
            raise ValueError('Text content is too short for analysis (min 50 chars)')
        return v

class CounterfactualRequest(BaseModel):
    content_id: str
    factor_name: str
    factor_value: float
    
    @validator('factor_value')
    def validate_factor_value(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Factor value must be between 0 and 1')
        return v

# Routes
@api_bp.route('/status', methods=['GET'])
def status():
    """Public endpoint to check API status."""
    return jsonify({
        'status': 'operational',
        'service': 'audience-intelligence-api',
        'models_loaded': model_manager.get_loaded_models(),
        'timestamp': datetime.now().isoformat()
    }), 200

@api_bp.route('/analyze', methods=['POST'])
def analyze_content():
    """Analyze content for audience engagement potential."""
    try:
        # Validate request
        data = request.get_json()
        content_request = ContentAnalysisRequest(**data)
        
        # Process primary image if provided
        image_tensor = None
        if content_request.primary_image:
            try:
                # Download image if it's a URL
                if content_request.primary_image.startswith('http'):
                    response = requests.get(content_request.primary_image)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                # Handle base64 encoded images
                elif content_request.primary_image.startswith('data:image'):
                    # Extract the base64 part
                    image_data = content_request.primary_image.split(',')[1]
                    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
                else:
                    raise ValueError("Unsupported image format")
                
                # Get image features
                image_tensor = model_manager.extract_visual_features(image)
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                # Continue without image
                pass
        
        # Get text features
        text_features = model_manager.extract_text_features(content_request.text)
        
        # Run fusion model
        if image_tensor is not None:
            prediction = model_manager.run_fusion_model(text_features, image_tensor)
        else:
            # Fall back to text-only prediction
            prediction = model_manager.run_text_only_prediction(text_features)
        
        # Extract causal factors
        causal_factors = model_manager.extract_causal_factors(
            text=content_request.text,
            content_features=prediction.get('content_features', None)
        )
        
        # Generate optimization suggestions
        suggestions = model_manager.generate_suggestions(
            causal_factors=causal_factors,
            text=content_request.text,
            title=content_request.title
        )
        
        # Create content ID for later reference
        content_id = model_manager.store_analysis_result(
            url=content_request.url,
            title=content_request.title,
            features=prediction
        )
        
        # Return results
        return jsonify({
            'content_id': content_id,
            'engagement_score': float(prediction['engagement_score']),
            'sentiment_score': float(prediction['sentiment_score']),
            'sentiment_category': prediction['sentiment_category'],
            'causal_factors': causal_factors,
            'suggestions': suggestions
        }), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Validation Error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            'error': 'Analysis Error',
            'message': 'An error occurred while analyzing content'
        }), 500

@api_bp.route('/causal-analysis', methods=['POST'])
def causal_analysis():
    """Perform detailed causal analysis on content."""
    try:
        data = request.get_json()
        content_request = ContentAnalysisRequest(**data)
        
        # Extract text features
        text_features = model_manager.extract_text_features(content_request.text)
        
        # Run causal analysis
        causal_graph = model_manager.generate_causal_graph(
            text=content_request.text,
            text_features=text_features
        )
        
        # Get causal factors and effects
        causal_factors = model_manager.extract_causal_factors(
            text=content_request.text,
            content_features=text_features  # Use text features as fallback
        )
        
        # Return results
        return jsonify({
            'causal_factors': causal_factors,
            'causal_graph': causal_graph,
            'analysis_time': datetime.now().isoformat()
        }), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Validation Error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Causal analysis error: {e}")
        return jsonify({
            'error': 'Analysis Error',
            'message': 'An error occurred during causal analysis'
        }), 500

@api_bp.route('/counterfactual', methods=['POST'])
def generate_counterfactual():
    """Generate counterfactual predictions."""
    try:
        data = request.get_json()
        cf_request = CounterfactualRequest(**data)
        
        # Generate counterfactual
        result = model_manager.generate_counterfactual(
            content_id=cf_request.content_id,
            factor_name=cf_request.factor_name,
            factor_value=cf_request.factor_value
        )
        
        # Return counterfactual results
        return jsonify({
            'original_score': result['original_score'],
            'counterfactual_score': result['counterfactual_score'],
            'difference': result['difference'],
            'factor_name': cf_request.factor_name,
            'factor_value': cf_request.factor_value
        }), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Validation Error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Counterfactual error: {e}")
        return jsonify({
            'error': 'Analysis Error',
            'message': 'An error occurred generating counterfactual'
        }), 500

@api_bp.route('/optimize', methods=['POST'])
def optimize_content():
    """Generate content optimization suggestions."""
    try:
        data = request.get_json()
        content_request = ContentAnalysisRequest(**data)
        
        # Extract text features
        text_features = model_manager.extract_text_features(content_request.text)
        
        # Generate optimization suggestions
        suggestions = model_manager.generate_optimization_suggestions(
            text=content_request.text,
            title=content_request.title,
            text_features=text_features
        )
        
        # Return suggestions
        return jsonify({
            'suggestions': suggestions,
            'title_suggestions': suggestions.get('title', []),
            'content_suggestions': suggestions.get('content', []),
            'engagement_delta': suggestions.get('estimated_improvement', 0)
        }), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Validation Error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return jsonify({
            'error': 'Analysis Error',
            'message': 'An error occurred generating optimization suggestions'
        }), 500 