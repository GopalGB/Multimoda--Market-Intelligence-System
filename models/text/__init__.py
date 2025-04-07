# models/text/__init__.py
"""
Text encoding and feature extraction components.

This module provides text processing with RoBERTa and specialized
text feature extractors for audience intelligence.
"""

from models.text.roberta_model import RoBERTaWrapper
from models.text.text_features import TextFeatureExtractor, TextFeatureTypes

__all__ = [
    'RoBERTaWrapper',
    'TextFeatureExtractor',
    'TextFeatureTypes'
]