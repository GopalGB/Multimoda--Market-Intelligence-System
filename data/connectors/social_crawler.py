# data/connectors/social_crawler.py
import pandas as pd
import requests
import datetime
from typing import Dict, Any, List, Optional
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor

class SocialMediaCrawler:
    """
    Crawler for social media platforms to collect data related to media content.
    Supports Twitter, Instagram, TikTok, and Facebook.
    """
    PLATFORMS = {
        "twitter": "https://api.twitter.com/2/",
        "instagram": "https://graph.instagram.com/v14.0/",
        "tiktok": "https://open-api.tiktok.com/v2/",
        "facebook": "https://graph.facebook.com/v13.0/",
    }
    
    def __init__(
        self,
        platforms: List[str] = ["twitter", "instagram"],
        api_keys: Optional[Dict[str, str]] = None,
        cache_dir: str = "./cache/social",
        max_workers: int = 4
    ):
        """
        Initialize the social media crawler.
        
        Args:
            platforms: List of platforms to crawl
            api_keys: Dictionary mapping platform names to API keys
            cache_dir: Directory for caching data
            max_workers: Maximum number of worker threads
        """
        self.platforms = [p for p in platforms if p in self.PLATFORMS]
        if not self.platforms:
            raise ValueError(f"No supported platforms selected. " +
                            f"Supported platforms: {', '.join(self.PLATFORMS.keys())}")
        
        # Initialize API keys
        self.api_keys = api_keys or {}
        for platform in self.platforms:
            if platform not in self.api_keys:
                env_key = f"{platform.upper()}_API_KEY"
                self.api_keys[platform] = os.environ.get(env_key)
                
                if not self.api_keys[platform]:
                    print(f"Warning: No API key provided for {platform}")
        
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def search_content_mentions(
        self,
        content_names: List[str],
        start_date: datetime.date,
        end_date: datetime.date,
        platforms: Optional[List[str]] = None,
        sentiment_analysis: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Search for mentions of content across social media platforms.
        
        Args:
            content_names: List of content names to search for
            start_date: Start date for the search
            end_date: End date for the search
            platforms: List of platforms to search (defaults to all configured platforms)
            sentiment_analysis: Whether to include sentiment analysis
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with social media mentions and metrics
        """
        # Use all configured platforms if none specified
        platforms = platforms or self.platforms
        platforms = [p for p in platforms if p in self.platforms]
        
        # Search each platform in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for platform in platforms:
                for content_name in content_names:
                    future = executor.submit(
                        self._search_single_platform,
                        platform=platform,
                        content_name=content_name,
                        start_date=start_date,
                        end_date=end_date,
                        sentiment_analysis=sentiment_analysis,
                        use_cache=use_cache
                    )
                    futures[future] = (platform, content_name)
            
            # Collect results
            for future in futures:
                platform, content_name = futures[future]
                try:
                    result = future.result()
                    if not result.empty:
                        results.append(result)
                except Exception as e:
                    print(f"Error fetching data for {content_name} on {platform}: {e}")
        
        # Combine results
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _search_single_platform(
        self,
        platform: str,
        content_name: str,
        start_date: datetime.date,
        end_date: datetime.date,
        sentiment_analysis: bool,
        use_cache: bool
    ) -> pd.DataFrame:
        """
        Search a single platform for mentions of a content name.
        
        Args:
            platform: Platform name
            content_name: Content name to search for
            start_date: Start date for the search
            end_date: End date for the search
            sentiment_analysis: Whether to include sentiment analysis
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with social media mentions
        """
        # Implementation for searching on a single platform
        # ...
        
        # Placeholder for demonstration
        return pd.DataFrame({
            "platform": [platform] * 3,
            "content_name": [content_name] * 3,
            "post_date": ["2022-01-15", "2022-01-16", "2022-01-17"],
            "post_text": [
                f"Just watched {content_name} and loved it!",
                f"{content_name} was amazing, can't wait for more!",
                f"Disappointed with {content_name}, not worth the hype."
            ],
            "user_id": ["user123", "user456", "user789"],
            "likes": [120, 85, 45],
            "shares": [30, 15, 5],
            "sentiment": [0.8, 0.9, -0.6] if sentiment_analysis else [None, None, None]
        })