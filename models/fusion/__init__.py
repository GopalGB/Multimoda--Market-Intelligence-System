# models/fusion/__init__.py
"""
Multimodal fusion models for combining visual and textual features.

This module contains the fusion model architecture and cross-attention mechanisms
for effective multimodal learning.
"""

from models.fusion.fusion_model import MultimodalFusionModel, FusionTypes
from models.fusion.cross_attention import CrossModalTransformer, CrossModalTransformerLayer

__all__ = [
    'MultimodalFusionModel',
    'FusionTypes',
    'CrossModalTransformer',
    'CrossModalTransformerLayer'
]