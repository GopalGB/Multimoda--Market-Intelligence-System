# models/__init__.py
"""
Cross-Modal Audience Intelligence Platform (CAIP) model components.

This package contains multimodal fusion models and feature extractors 
for analyzing content and predicting audience engagement.
"""

from models.fusion.fusion_model import MultimodalFusionModel
from models.visual.clip_model import CLIPWrapper
from models.text.roberta_model import RoBERTaWrapper
from models.optimization.onnx_export import ONNXExporter
from models.optimization.quantization import ModelQuantizer
from models.optimization.distillation import KnowledgeDistillation

__all__ = [
    'MultimodalFusionModel',
    'CLIPWrapper',
    'RoBERTaWrapper',
    'ONNXExporter',
    'ModelQuantizer',
    'KnowledgeDistillation'
]