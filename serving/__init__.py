# serving/__init__.py
"""
Deployment and serving components for audience intelligence models.

This package provides API endpoints, Ray Serve deployment,
and Kubernetes configuration for production deployments.
"""

from serving.api import app, main as api_main
from serving.ray_serve import create_deployment, main as ray_main

__all__ = [
    'app',
    'api_main',
    'create_deployment',
    'ray_main'
]