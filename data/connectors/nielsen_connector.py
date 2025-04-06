# data/connectors/nielsen_connector.py
import pandas as pd
import requests
from typing import Dict, Any, Optional, List
import datetime
import os
import json

class NielsenConnector:
    """
    Connector for Nielsen's proprietary panel data.
    Handles authentication, data fetching, and preprocessing.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://api.nielsen.com/data",
        cache_dir: str = "./cache"
    ):
        """
        Initialize the Nielsen connector.
        
        Args:
            api_key: Nielsen API key (if None, uses NIELSEN_API_KEY env variable)
            endpoint: API endpoint
            cache_dir: Directory for caching data
        """
        self.api_key = api_key or os.environ.get("NIELSEN_API_KEY")
        self.endpoint = endpoint
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def fetch_panel_data(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        metrics: List[str],
        demographics: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch Nielsen panel data.
        
        Args:
            start_date: Start date for the data
            end_date: End date for the data
            metrics: List of metrics to fetch
            demographics: Optional demographic filters
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with Nielsen panel data
        """
        # Check cache first if use_cache is True
        cache_file = self._get_cache_file_path(
            start_date, end_date, metrics, demographics
        )
        
        if use_cache and os.path.exists(cache_file):
            return pd.read_csv(cache_file)
        
        # Prepare request parameters
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "metrics": ",".join(metrics),
            "format": "json"
        }
        
        # Add demographics if provided
        if demographics:
            params["demographics"] = json.dumps(demographics)
        
        # Make API request
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(
            f"{self.endpoint}/panel", 
            headers=headers,
            params=params
        )
        
        # Raise error if request failed
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data["data"])
        
        # Cache the data if use_cache is True
        if use_cache:
            df.to_csv(cache_file, index=False)
        
        return df
    
    def fetch_engagement_metrics(
        self,
        content_ids: List[str],
        metrics: List[str] = ["completion_rate", "share_rate", "like_rate"],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch engagement metrics for specific content.
        
        Args:
            content_ids: List of content IDs
            metrics: List of engagement metrics to fetch
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with engagement metrics
        """
        # Similar implementation to fetch_panel_data, but for engagement metrics
        # ...
        
        # Placeholder for demonstration
        return pd.DataFrame({
            "content_id": content_ids,
            "completion_rate": [0.85, 0.72, 0.91],
            "share_rate": [0.12, 0.08, 0.15],
            "like_rate": [0.25, 0.18, 0.31]
        })
    
    def _get_cache_file_path(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        metrics: List[str],
        demographics: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a cache file path based on request parameters."""
        # Create a hash of the request parameters
        param_str = f"{start_date}-{end_date}-{'-'.join(metrics)}"
        if demographics:
            param_str += f"-{json.dumps(demographics, sort_keys=True)}"
        
        import hashlib
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        return os.path.join(self.cache_dir, f"nielsen_data_{param_hash}.csv")