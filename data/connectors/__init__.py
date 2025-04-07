# data/connectors/__init__.py
"""
Data connectors for various content and audience sources.

This module provides interfaces to Nielsen panel data, streaming platforms,
and social media content.
"""

from data.connectors.nielsen_connector import NielsenConnector
from data.connectors.streaming_api import StreamingPlatformConnector
from data.connectors.social_crawler import SocialMediaCrawler

__all__ = [
    'NielsenConnector',
    'StreamingPlatformConnector',
    'SocialMediaCrawler'
]