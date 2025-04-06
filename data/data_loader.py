# data/data_loader.py
import pandas as pd
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .connectors.nielsen_connector import NielsenConnector
from .connectors.streaming_api import StreamingPlatformConnector
from .connectors.social_crawler import SocialMediaCrawler

class DataLoader:
    """
    Unified data loader for audience intelligence platform.
    Loads and combines data from multiple sources.
    """
    def __init__(
        self,
        nielsen_connector: Optional[NielsenConnector] = None,
        streaming_connectors: Optional[Dict[str, StreamingPlatformConnector]] = None,
        social_crawler: Optional[SocialMediaCrawler] = None,
        cache_dir: str = "./cache/combined",
        max_workers: int = 4
    ):
        """
        Initialize the data loader.
        
        Args:
            nielsen_connector: Nielsen connector instance
            streaming_connectors: Dict mapping platform names to connector instances
            social_crawler: Social media crawler instance
            cache_dir: Directory for caching combined data
            max_workers: Maximum number of worker threads
        """
        # Initialize connectors if not provided
        self.nielsen_connector = nielsen_connector or NielsenConnector()
        
        if streaming_connectors is None:
            self.streaming_connectors = {
                "netflix": StreamingPlatformConnector("netflix")
            }
        else:
            self.streaming_connectors = streaming_connectors
            
        self.social_crawler = social_crawler or SocialMediaCrawler()
        
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_audience_data(
        self,
        content_ids: List[str],
        content_names: List[str],
        start_date: datetime.date,
        end_date: datetime.date,
        metrics: List[str] = ["views", "completion_rate", "engagement"],
        demographics: Optional[Dict[str, Any]] = None,
        platforms: Optional[List[str]] = None,
        include_social: bool = True,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load audience data from multiple sources.
        
        Args:
            content_ids: List of content IDs
            content_names: List of content names (for social media)
            start_date: Start date for the data
            end_date: End date for the data
            metrics: List of metrics to fetch
            demographics: Optional demographic filters
            platforms: List of streaming platforms to include
            include_social: Whether to include social media data
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping data types to DataFrames
        """
        # Check cache first if use_cache is True
        cache_file = self._get_cache_file_path(
            content_ids, content_names, start_date, end_date, 
            metrics, demographics, platforms, include_social
        )
        
        if use_cache and os.path.exists(cache_file):
            # Load from cache
            data_dict = {}
            with pd.HDFStore(cache_file, 'r') as store:
                for key in store.keys():
                    # Remove leading '/'
                    clean_key = key[1:]
                    data_dict[clean_key] = store[key]
            
            return data_dict
        
        # Load data in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            futures = {}
            
            # Nielsen panel data
            nielsen_future = executor.submit(
                self.nielsen_connector.fetch_panel_data,
                start_date=start_date,
                end_date=end_date,
                metrics=metrics,
                demographics=demographics,
                use_cache=use_cache
            )
            futures["nielsen"] = nielsen_future
            
            # Streaming platform data
            streaming_futures = {}
            for platform, connector in self.streaming_connectors.items():
                if platforms is None or platform in platforms:
                    future = executor.submit(
                        connector.fetch_audience_metrics,
                        content_ids=content_ids,
                        start_date=start_date,
                        end_date=end_date,
                        metrics=metrics,
                        use_cache=use_cache
                    )
                    streaming_futures[platform] = future
            
            # Social media data
            if include_social:
                social_future = executor.submit(
                    self.social_crawler.search_content_mentions,
                    content_names=content_names,
                    start_date=start_date,
                    end_date=end_date,
                    sentiment_analysis=True,
                    use_cache=use_cache
                )
                futures["social"] = social_future
            
            # Collect results
            data_dict = {}
            
            # Nielsen data
            try:
                data_dict["nielsen"] = nielsen_future.result()
            except Exception as e:
                print(f"Error fetching Nielsen data: {e}")
                data_dict["nielsen"] = pd.DataFrame()
            
            # Streaming data
            streaming_data = {}
            for platform, future in streaming_futures.items():
                try:
                    streaming_data[platform] = future.result()
                except Exception as e:
                    print(f"Error fetching {platform} data: {e}")
                    streaming_data[platform] = pd.DataFrame()
            
            # Combine streaming data
            if streaming_data:
                dfs = []
                for platform, df in streaming_data.items():
                    if not df.empty:
                        df["platform"] = platform
                        dfs.append(df)
                
                if dfs:
                    data_dict["streaming"] = pd.concat(dfs, ignore_index=True)
                else:
                    data_dict["streaming"] = pd.DataFrame()
            else:
                data_dict["streaming"] = pd.DataFrame()
            
            # Social data
            if include_social:
                try:
                    data_dict["social"] = social_future.result()
                except Exception as e:
                    print(f"Error fetching social media data: {e}")
                    data_dict["social"] = pd.DataFrame()
        
        # Cache the combined data if use_cache is True
        if use_cache:
            with pd.HDFStore(cache_file, 'w') as store:
                for key, df in data_dict.items():
                    if not df.empty:
                        store[key] = df
        
        return data_dict
    
    def _get_cache_file_path(
        self,
        content_ids: List[str],
        content_names: List[str],
        start_date: datetime.date,
        end_date: datetime.date,
        metrics: List[str],
        demographics: Optional[Dict[str, Any]],
        platforms: Optional[List[str]],
        include_social: bool
    ) -> str:
        """Generate a cache file path based on request parameters."""
        # Create a hash of the request parameters
        param_str = f"{'-'.join(content_ids)}-{'-'.join(content_names)}-"
        param_str += f"{start_date}-{end_date}-{'-'.join(metrics)}"
        
        if demographics:
            import json
            param_str += f"-{json.dumps(demographics, sort_keys=True)}"
            
        if platforms:
            param_str += f"-{'-'.join(platforms)}"
            
        if include_social:
            param_str += "-social"
        
        import hashlib
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        return os.path.join(self.cache_dir, f"combined_data_{param_hash}.h5")