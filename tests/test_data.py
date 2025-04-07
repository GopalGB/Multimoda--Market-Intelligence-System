# tests/test_data.py
import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock

from data.data_loader import DataLoader
from data.connectors.social_crawler import SocialMediaCrawler
from data.connectors.nielsen_connector import NielsenConnector
from data.connectors.streaming_api import StreamingPlatformConnector
from data.preprocessing.text_preprocessor import TextPreprocessor
from data.preprocessing.image_preprocessor import ImagePreprocessor
from data.preprocessing.feature_engineering import FeatureEngineer


class TestDataConnectors(unittest.TestCase):
    """Test data connector components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_cache_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temp directories
        import shutil
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    def test_social_crawler_initialization(self):
        """Test SocialMediaCrawler initialization."""
        # Test with defaults
        crawler = SocialMediaCrawler(cache_dir=self.test_cache_dir)
        self.assertEqual(crawler.platforms, ["twitter", "instagram"])
        
        # Test with custom platforms
        crawler = SocialMediaCrawler(
            platforms=["twitter", "tiktok"],
            cache_dir=self.test_cache_dir
        )
        self.assertEqual(crawler.platforms, ["twitter", "tiktok"])
        
        # Test with invalid platforms
        with self.assertRaises(ValueError):
            SocialMediaCrawler(platforms=["invalid_platform"], cache_dir=self.test_cache_dir)
    
    @patch('requests.get')
    def test_nielsen_connector_fetch_panel_data(self, mock_get):
        """Test NielsenConnector fetch_panel_data method."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"date": "2023-01-01", "metric": "views", "value": 1000}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Create connector and fetch data
        connector = NielsenConnector(api_key="test_key", cache_dir=self.test_cache_dir)
        data = connector.fetch_panel_data(
            start_date=pd.Timestamp("2023-01-01").date(),
            end_date=pd.Timestamp("2023-01-31").date(),
            metrics=["views", "engagement"]
        )
        
        # Assert data was fetched correctly
        self.assertIsInstance(data, pd.DataFrame)
        mock_get.assert_called_once()
    
    def test_streaming_platform_connector(self):
        """Test StreamingPlatformConnector."""
        # Test initialization with valid platform
        connector = StreamingPlatformConnector("netflix", cache_dir=self.test_cache_dir)
        self.assertEqual(connector.platform, "netflix")
        
        # Test initialization with invalid platform
        with self.assertRaises(ValueError):
            StreamingPlatformConnector("invalid_platform", cache_dir=self.test_cache_dir)
        
        # Test rate limiting mechanism
        connector = StreamingPlatformConnector("netflix", rate_limit=10, cache_dir=self.test_cache_dir)
        initial_time = connector.last_request_time
        connector._manage_rate_limit()
        self.assertGreater(connector.last_request_time, initial_time)


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing components."""
    
    def test_text_preprocessor(self):
        """Test TextPreprocessor functionality."""
        preprocessor = TextPreprocessor(remove_stopwords=True)
        
        # Test cleaning
        dirty_text = "This is a test URL: https://example.com with HTML <tags> and numbers 12345"
        clean_text = preprocessor.clean_text(dirty_text)
        self.assertNotIn("https://example.com", clean_text)
        self.assertNotIn("<tags>", clean_text)
        self.assertNotIn("12345", clean_text)
        
        # Test tokenization
        tokens = preprocessor.tokenize("This is a simple test sentence.")
        self.assertIsInstance(tokens, list)
        # Stopwords should be removed
        self.assertNotIn("is", tokens)
        self.assertNotIn("a", tokens)
    
    def test_image_preprocessor(self):
        """Test ImagePreprocessor functionality."""
        preprocessor = ImagePreprocessor(target_size=(224, 224), normalize=True)
        
        # Create a mock image
        import numpy as np
        from PIL import Image
        mock_image = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype('uint8'))
        
        # Test preprocessing
        tensor = preprocessor.preprocess(mock_image)
        self.assertEqual(tensor.shape[1:], (3, 224, 224))  # Check shape (C,H,W)
    
    def test_feature_engineer(self):
        """Test FeatureEngineer functionality."""
        # Create sample data
        data = pd.DataFrame({
            'numeric_feat1': [1.0, 2.0, 3.0, 4.0],
            'numeric_feat2': [5.0, 6.0, 7.0, 8.0],
            'category': ['A', 'B', 'A', 'B'],
            'date_col': pd.date_range(start='2023-01-01', periods=4)
        })
        
        # Initialize feature engineer
        engineer = FeatureEngineer(scaler="standard")
        
        # Test fitting
        engineer.fit(data, target_col=None)
        self.assertIn('numeric_feat1', engineer.numerical_features)
        self.assertIn('numeric_feat2', engineer.numerical_features)
        self.assertIn('category', engineer.categorical_features)
        
        # Test temporal feature extraction
        result = engineer.add_temporal_features(data, date_columns=['date_col'])
        self.assertIn('date_col_year', result.columns)
        self.assertIn('date_col_month', result.columns)
        self.assertIn('date_col_day', result.columns)


class TestDataLoader(unittest.TestCase):
    """Test the DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_cache_dir = tempfile.mkdtemp()
        
        # Create mock connectors
        self.mock_nielsen = MagicMock(spec=NielsenConnector)
        self.mock_nielsen.fetch_panel_data.return_value = pd.DataFrame({
            "date": pd.date_range(start="2023-01-01", periods=3),
            "metric": ["views", "engagement", "completion"],
            "value": [1000, 0.75, 0.80]
        })
        
        self.mock_streaming = MagicMock(spec=StreamingPlatformConnector)
        self.mock_streaming.fetch_audience_metrics.return_value = pd.DataFrame({
            "content_id": ["A", "B", "C"],
            "views": [10000, 8500, 25000],
            "average_watch_time": [42, 38, 95],
            "completion_rate": [0.75, 0.62, 0.88]
        })
        
        self.mock_social = MagicMock(spec=SocialMediaCrawler)
        self.mock_social.search_content_mentions.return_value = pd.DataFrame({
            "platform": ["twitter", "instagram", "twitter"],
            "content_name": ["Show A", "Show A", "Show B"],
            "post_date": ["2023-01-15", "2023-01-16", "2023-01-17"],
            "sentiment": [0.8, 0.9, -0.6]
        })
        
        # Create DataLoader with mock connectors
        self.data_loader = DataLoader(
            nielsen_connector=self.mock_nielsen,
            streaming_connectors={"netflix": self.mock_streaming},
            social_crawler=self.mock_social,
            cache_dir=self.test_cache_dir
        )
        
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    def test_load_audience_data(self):
        """Test load_audience_data method."""
        # Call method
        result = self.data_loader.load_audience_data(
            content_ids=["A", "B"],
            content_names=["Show A", "Show B"],
            start_date=pd.Timestamp("2023-01-01").date(),
            end_date=pd.Timestamp("2023-01-31").date(),
            metrics=["views", "engagement"],
            include_social=True,
            use_cache=False
        )
        
        # Verify results
        self.assertIsInstance(result, dict)
        self.assertIn("nielsen", result)
        self.assertIn("streaming", result)
        self.assertIn("social", result)
        
        # Check that connectors were called with correct parameters
        self.mock_nielsen.fetch_panel_data.assert_called_once()
        self.mock_streaming.fetch_audience_metrics.assert_called_once()
        self.mock_social.search_content_mentions.assert_called_once()


if __name__ == '__main__':
    unittest.main()