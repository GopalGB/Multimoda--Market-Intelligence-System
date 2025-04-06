# data/connectors/streaming_api.py
import pandas as pd
import requests
import datetime
from typing import Dict, Any, List, Optional
import os
import json
import time

class StreamingPlatformConnector:
    """
    Connector for streaming platform APIs.
    Supports multiple platforms including Netflix, Hulu, and Disney+.
    """
    PLATFORMS = {
        "netflix": "https://api.netflix.com/v2/",
        "hulu": "https://api.hulu.com/v1/",
        "disney": "https://api.disneyplus.com/v1/",
        "amazon": "https://api.amazonprimevideo.com/v1/",
    }
    
    def __init__(
        self,
        platform: str,
        api_key: Optional[str] = None,
        rate_limit: int = 60,  # requests per minute
        cache_dir: str = "./cache/streaming"
    ):
        """
        Initialize the streaming platform connector.
        
        Args:
            platform: Platform name (netflix, hulu, disney, amazon)
            api_key: API key (if None, uses env variable)
            rate_limit: Rate limit in requests per minute
            cache_dir: Directory for caching data
        """
        if platform not in self.PLATFORMS:
            raise ValueError(f"Unsupported platform: {platform}. " +
                            f"Supported platforms: {', '.join(self.PLATFORMS.keys())}")
        
        self.platform = platform
        self.base_url = self.PLATFORMS[platform]
        self.api_key = api_key or os.environ.get(f"{platform.upper()}_API_KEY")
        self.rate_limit = rate_limit
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Request tracking for rate limiting
        self.last_request_time = 0
    
    def fetch_content_metadata(
        self,
        content_ids: List[str],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch metadata for specific content.
        
        Args:
            content_ids: List of content IDs
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with content metadata
        """
        # Implementation for fetching content metadata
        # ...
        
        # Placeholder for demonstration
        return pd.DataFrame({
            "content_id": content_ids,
            "title": ["Show A", "Show B", "Movie C"],
            "type": ["series", "series", "movie"],
            "genre": ["drama", "comedy", "action"],
            "release_date": ["2021-01-15", "2020-11-20", "2022-03-05"]
        })
    
    def fetch_audience_metrics(
        self,
        content_ids: List[str],
        start_date: datetime.date,
        end_date: datetime.date,
        metrics: List[str] = ["views", "average_watch_time", "completion_rate"],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch audience metrics for specific content.
        
        Args:
            content_ids: List of content IDs
            start_date: Start date for the data
            end_date: End date for the data
            metrics: List of metrics to fetch
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with audience metrics
        """
        # Implementation for fetching audience metrics
        # ...
        
        # Placeholder for demonstration
        return pd.DataFrame({
            "content_id": content_ids,
            "views": [120000, 85000, 250000],
            "average_watch_time": [42, 38, 95],
            "completion_rate": [0.75, 0.62, 0.88]
        })
    
    def _manage_rate_limit(self):
        """Manage API rate limiting."""
        current_time = time.time()
        elapsed_time = current_time - self.last_request_time
        
        # If we're making requests too quickly, wait
        min_interval = 60 / self.rate_limit  # seconds per request
        if elapsed_time < min_interval:
            sleep_time = min_interval - elapsed_time
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()