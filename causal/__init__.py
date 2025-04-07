# causal/__init__.py
"""
Causal inference components for audience intelligence.

This package provides causal discovery, structural causal models, 
counterfactual analysis, and causal feature selection capabilities.
"""

from causal.structural_model import CausalGraph, StructuralCausalModel
from causal.causal_features import CausalFeatureSelector
from causal.counterfactual import CounterfactualAnalyzer

__all__ = [
    'CausalGraph',
    'StructuralCausalModel',
    'CausalFeatureSelector',
    'CounterfactualAnalyzer'
]