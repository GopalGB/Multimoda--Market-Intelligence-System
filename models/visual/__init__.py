# models/visual/__init__.py
"""
Visual encoding and feature extraction components.

This module provides image processing with CLIP and specialized
visual feature extractors for audience intelligence.
"""

from models.visual.clip_model import CLIPWrapper
from models.visual.visual_features import VisualFeatureExtractor, VisualFeatureTypes

__all__ = [
    'CLIPWrapper',
    'VisualFeatureExtractor',
    'VisualFeatureTypes'
]