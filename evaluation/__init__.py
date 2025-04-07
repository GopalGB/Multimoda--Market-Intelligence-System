# evaluation/__init__.py
"""
Evaluation metrics and benchmarking tools for audience intelligence models.

This package provides specialized metrics, performance analysis,
ablation studies, and model benchmarking capabilities.
"""

from evaluation.metrics import EngagementMetrics, CausalMetrics, MultimodalMetrics
from evaluation.benchmark import FusionModelBenchmark, CausalModelBenchmark, BenchmarkDataset
from evaluation.ablation import AblationStudy, compare_ablation_studies

__all__ = [
    'EngagementMetrics',
    'CausalMetrics', 
    'MultimodalMetrics',
    'FusionModelBenchmark',
    'CausalModelBenchmark',
    'BenchmarkDataset',
    'AblationStudy',
    'compare_ablation_studies'
]