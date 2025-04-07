# tests/test_serving.py
import unittest
import json
import io
import base64
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import tempfile
import os
import torch
import numpy as np
from PIL import Image

# Import the API module (patch the model loaders)
with patch('serving.api.load_fusion_model'), \
     patch('serving.api.load_causal_model'), \
     patch('serving.api.CLIPWrapper'), \
     patch('serving.api.RoBERTaWrapper'):
    from serving.api import app

# Create test client
client = TestClient(app)


class TestAPIEndpoints(unittest.TestCase):
    """Test the API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock fusion model
        self.mock_fusion_model = MagicMock()
        self.mock_fusion_model.predict_engagement.return_value = {
            "engagement_score": [0.75],
            "expected_views": 10000,
            "expected_shares": 500
        }
        self.mock_fusion_model.predict_sentiment.return_value = {
            "sentiment_score": [0.8],
            "sentiment_label": ["positive"]
        }
        
        # Create a mock causal model
        self.mock_causal_model = MagicMock()
        self.mock_causal_model.estimate_all_effects.return_value = {
            "feature1": {"causal_effect": 0.3, "p_value": 0.01},
            "feature2": {"causal_effect": 0.5, "p_value": 0.001}
        }
        
        # Create a mock clip model
        self.mock_clip = MagicMock()
        self.mock_clip.encode_images.return_value = torch.zeros((1, 768))
        
        # Create a mock roberta model
        self.mock_roberta = MagicMock()
        self.mock_roberta.encode_text.return_value = torch.zeros((1, 768))
        
        # Patch the app to use our mock models
        app.dependency_overrides[app.__dict__["get_models"]] = lambda: {
            "fusion_model": self.mock_fusion_model,
            "clip_model": self.mock_clip,
            "roberta_model": self.mock_roberta,
            "causal_model": self.mock_causal_model,
            "model_versions": {}
        }
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        
        # Should return 200 OK
        self.assertEqual(response.status_code, 200)
        
        # Should have status field
        data = response.json()
        self.assertEqual(data["status"], "ok")
    
    def test_engagement_prediction_endpoint(self):
        """Test the engagement prediction endpoint."""
        # Create request body
        request_data = {
            "text_content": "This is a test post about a new TV show",
            "visual_content_url": "https://example.com/image.jpg"
        }
        
        # Mock requests.get for downloading images
        with patch('requests.get') as mock_get:
            # Create a mock response with image content
            mock_response = MagicMock()
            mock_response.status_code = 200
            
            # Create a small test image
            test_image = Image.new('RGB', (100, 100), color='red')
            img_byte_arr = io.BytesIO()
            test_image.save(img_byte_arr, format='JPEG')
            mock_response.content = img_byte_arr.getvalue()
            
            mock_get.return_value = mock_response
            
            # Make the request
            response = client.post(f"/api/v1/predict/engagement", json=request_data)
        
        # Should return 200 OK
        self.assertEqual(response.status_code, 200)
        
        # Should have engagement prediction
        data = response.json()
        self.assertIn("engagement_prediction", data)
        self.assertIn("processing_time_ms", data)
        
        # Models should have been called
        self.mock_roberta.encode_text.assert_called_once()
        self.mock_clip.encode_images.assert_called_once()
        self.mock_fusion_model.predict_engagement.assert_called_once()
    
    def test_sentiment_analysis_endpoint(self):
        """Test the sentiment analysis endpoint."""
        # Create request body
        request_data = {
            "text_content": "This is a very positive post about a great TV show!",
            "visual_content_url": None
        }
        
        # Make the request
        response = client.post(f"/api/v1/analyze/sentiment", json=request_data)
        
        # Should return 200 OK
        self.assertEqual(response.status_code, 200)
        
        # Should have sentiment analysis
        data = response.json()
        self.assertIn("sentiment_analysis", data)
        self.assertIn("processing_time_ms", data)
        
        # Models should have been called
        self.mock_roberta.encode_text.assert_called_once()
        self.mock_fusion_model.predict_sentiment.assert_called_once()
    
    def test_causal_analysis_endpoint(self):
        """Test the causal analysis endpoint."""
        # Create request body
        request_data = {
            "content_features": {
                "feature1": 0.5,
                "feature2": 0.8,
                "feature3": 0.3
            },
            "target_engagement": 0.9
        }
        
        # Make the request
        response = client.post(f"/api/v1/analyze/causal", json=request_data)
        
        # Should return 200 OK
        self.assertEqual(response.status_code, 200)
        
        # Should have causal factors
        data = response.json()
        self.assertIn("causal_factors", data)
        self.assertIn("processing_time_ms", data)
        
        # Causal model should have been called
        self.mock_causal_model.estimate_all_effects.assert_called_once()
    
    def test_form_file_upload(self):
        """Test file upload via form data."""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='blue')
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Make the request with form data
        response = client.post(
            f"/api/v1/predict/engagement",
            files={"image": ("test.jpg", img_byte_arr, "image/jpeg")},
            data={"text": "This is a test post with an uploaded image"}
        )
        
        # Should return 200 OK
        self.assertEqual(response.status_code, 200)
        
        # Should have engagement prediction
        data = response.json()
        self.assertIn("engagement_prediction", data)
        self.assertTrue(data["image_analyzed"])


class TestRayServe(unittest.TestCase):
    """Test the Ray Serve deployment (light testing, mocked implementation)."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Patch ray and ray.serve
        self.patcher1 = patch('ray.init')
        self.patcher2 = patch('ray.serve.start')
        self.patcher3 = patch('ray.serve.run')
        
        # Start patches
        self.mock_ray_init = self.patcher1.start()
        self.mock_serve_start = self.patcher2.start()
        self.mock_serve_run = self.patcher3.start()
        
        # Patch model loading functions
        self.patcher4 = patch('serving.ray_serve.load_fusion_model')
        self.patcher5 = patch('serving.ray_serve.load_clip_model')
        self.patcher6 = patch('serving.ray_serve.load_roberta_model')
        self.patcher7 = patch('serving.ray_serve.load_causal_model')
        
        self.mock_load_fusion = self.patcher4.start()
        self.mock_load_clip = self.patcher5.start()
        self.mock_load_roberta = self.patcher6.start()
        self.mock_load_causal = self.patcher7.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()
        self.patcher6.stop()
        self.patcher7.stop()
    
    def test_ray_serve_initialization(self):
        """Test Ray Serve initialization."""
        # Import and run the module
        with patch('serving.ray_serve.time.sleep', side_effect=KeyboardInterrupt):
            try:
                from serving.ray_serve import main
                main()
            except KeyboardInterrupt:
                pass
        
        # Should call ray.init
        self.mock_ray_init.assert_called_once()
        
        # Should start the Ray Serve instance
        self.mock_serve_start.assert_called_once()
        
        # Should run the deployment
        self.mock_serve_run.assert_called_once()


class TestKubernetesConfigs(unittest.TestCase):
    """Test Kubernetes configuration files."""
    
    def test_deployment_yaml(self):
        """Test the Kubernetes deployment YAML file."""
        import yaml
        
        # Load the deployment yaml
        try:
            with open("serving/kubernetes/deployment.yaml", "r") as f:
                deployment = yaml.safe_load_all(f)
                configs = list(deployment)
        except FileNotFoundError:
            self.skipTest("Kubernetes deployment file not found")
        
        # Should have at least one config (the Deployment)
        self.assertGreaterEqual(len(configs), 1)
        
        # Check deployment spec
        deployment = next((c for c in configs if c["kind"] == "Deployment"), None)
        
        if deployment:
            self.assertEqual(deployment["metadata"]["name"], "caip-audience-intelligence")
            self.assertIn("replicas", deployment["spec"])
            
            # Check container spec
            containers = deployment["spec"]["template"]["spec"]["containers"]
            self.assertGreaterEqual(len(containers), 1)
            
            # Check resource requests and limits
            resources = containers[0]["resources"]
            self.assertIn("requests", resources)
            self.assertIn("limits", resources)
    
    def test_service_yaml(self):
        """Test the Kubernetes service YAML file."""
        import yaml
        
        # Load the service yaml
        try:
            with open("serving/kubernetes/service.yaml", "r") as f:
                service = yaml.safe_load(f)
        except FileNotFoundError:
            self.skipTest("Kubernetes service file not found")
        
        # Check service spec
        self.assertEqual(service["kind"], "Service")
        self.assertEqual(service["metadata"]["name"], "caip-audience-intelligence")
        
        # Check ports
        ports = service["spec"]["ports"]
        self.assertGreaterEqual(len(ports), 1)
        
        # Check selector
        self.assertIn("selector", service["spec"])
        self.assertEqual(service["spec"]["selector"]["app"], "caip-audience-intelligence")


if __name__ == '__main__':
    unittest.main()