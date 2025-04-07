# tests/test_models.py
import unittest
import numpy as np
import pandas as pd
import torch
import os
import tempfile
from unittest.mock import patch, MagicMock

from models.fusion.fusion_model import MultimodalFusionModel
from models.visual.clip_model import CLIPWrapper
from models.text.roberta_model import RoBERTaWrapper
from models.optimization.onnx_export import ONNXExporter
from models.optimization.quantization import ModelQuantizer


class TestTextModel(unittest.TestCase):
    """Test the text model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip actual model loading in tests
        self.patcher = patch("transformers.AutoModel.from_pretrained")
        self.patcher2 = patch("transformers.AutoTokenizer.from_pretrained")
        
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.ones((1, 10), dtype=torch.long),
            "attention_mask": torch.ones((1, 10), dtype=torch.long)
        })
        
        self.mock_model = self.patcher.start()
        self.mock_model.return_value = mock_model
        
        self.mock_tokenizer = self.patcher2.start()
        self.mock_tokenizer.return_value = mock_tokenizer
        
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        self.patcher2.stop()
    
    def test_roberta_model_initialization(self):
        """Test RoBERTa model initialization."""
        # Test initialization
        model = RoBERTaWrapper(model_name="roberta-base")
        self.assertEqual(model.model_name, "roberta-base")
        self.assertEqual(model.device, "cpu")  # Default to CPU in tests
        
        # Test initialization with specific device
        model = RoBERTaWrapper(model_name="roberta-base", device="cuda:0")
        self.assertEqual(model.device, "cuda:0")
    
    def test_encode_text(self):
        """Test text encoding functionality."""
        model = RoBERTaWrapper(model_name="roberta-base")
        
        # Mock the model output
        last_hidden_state = torch.rand(1, 10, 768)
        pooler_output = torch.rand(1, 768)
        
        model_output = MagicMock()
        model_output.last_hidden_state = last_hidden_state
        model_output.pooler_output = pooler_output
        
        model.model = MagicMock()
        model.model.return_value = model_output
        
        # Test single text encoding
        embeddings = model.encode_text("This is a test text")
        self.assertIsInstance(embeddings, torch.Tensor)
        
        # Test batch encoding
        batch_embeddings = model.encode_text(["Text 1", "Text 2"])
        self.assertIsInstance(batch_embeddings, torch.Tensor)


class TestVisualModel(unittest.TestCase):
    """Test the visual model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip actual model loading in tests
        self.patcher = patch("transformers.CLIPModel.from_pretrained")
        self.patcher2 = patch("transformers.CLIPProcessor.from_pretrained")
        
        # Mock the model and processor
        mock_model = MagicMock()
        mock_model.config.projection_dim = 768
        mock_model.return_value = mock_model
        
        mock_processor = MagicMock()
        mock_processor.return_value = mock_processor
        mock_processor.__call__ = MagicMock(return_value={
            "pixel_values": torch.rand((1, 3, 224, 224)),
            "attention_mask": torch.ones((1, 1), dtype=torch.long)
        })
        
        self.mock_model = self.patcher.start()
        self.mock_model.return_value = mock_model
        
        self.mock_processor = self.patcher2.start()
        self.mock_processor.return_value = mock_processor
        
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        self.patcher2.stop()
    
    def test_clip_model_initialization(self):
        """Test CLIP model initialization."""
        # Test initialization
        model = CLIPWrapper(model_name="openai/clip-vit-base-patch32")
        self.assertEqual(model.model_name, "openai/clip-vit-base-patch32")
        self.assertEqual(model.device, "cpu")  # Default to CPU in tests
        
        # Test initialization with specific device
        model = CLIPWrapper(model_name="openai/clip-vit-base-patch32", device="cuda:0")
        self.assertEqual(model.device, "cuda:0")
    
    def test_encode_images(self):
        """Test image encoding functionality."""
        model = CLIPWrapper(model_name="openai/clip-vit-base-patch32")
        
        # Mock the model output
        image_embeddings = torch.rand(1, 768)
        
        model_output = MagicMock()
        model_output.image_embeds = image_embeddings
        
        model.model = MagicMock()
        model.model.get_image_features = MagicMock(return_value=image_embeddings)
        
        # Create a test image (small black square)
        from PIL import Image
        test_image = Image.new('RGB', (50, 50), color='black')
        
        # Test single image encoding
        embeddings = model.encode_images(test_image)
        self.assertIsInstance(embeddings, torch.Tensor)
        
        # Test batch encoding (not implemented but should handle gracefully)
        with self.assertRaises(NotImplementedError):
            batch_embeddings = model.encode_images([test_image, test_image])


class TestFusionModel(unittest.TestCase):
    """Test the multimodal fusion model."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal fusion model for testing
        self.model = MultimodalFusionModel(
            visual_embedding_dim=768,
            text_embedding_dim=768,
            fusion_dim=512,
            output_dim=1,
            device="cpu"
        )
        
        # Create test inputs
        self.visual_features = torch.rand(2, 768)
        self.text_features = torch.rand(2, 768)
    
    def test_fusion_model_initialization(self):
        """Test fusion model initialization."""
        self.assertEqual(self.model.device, "cpu")
        self.assertEqual(self.model.fusion_dim, 512)
        self.assertEqual(self.model.output_dim, 1)
    
    def test_forward_pass(self):
        """Test model forward pass."""
        # Forward pass
        output = self.model(self.visual_features, self.text_features)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 1))
    
    def test_predict_engagement(self):
        """Test engagement prediction."""
        engagement = self.model.predict_engagement(self.visual_features, self.text_features)
        
        # Check that we get numeric predictions
        self.assertIsInstance(engagement, dict)
        self.assertIn("engagement_score", engagement)
        
        # Scores should be between 0 and 1
        scores = engagement["engagement_score"]
        self.assertTrue(all(0 <= score <= 1 for score in scores))
    
    def test_predict_sentiment(self):
        """Test sentiment prediction."""
        sentiment = self.model.predict_sentiment(self.visual_features, self.text_features)
        
        # Check that we get sentiment predictions
        self.assertIsInstance(sentiment, dict)
        self.assertIn("sentiment_score", sentiment)
        self.assertIn("sentiment_label", sentiment)
    
    def test_save_and_load(self):
        """Test model saving and loading."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "fusion_model.pt")
            
            # Save the model
            self.model.save(model_path)
            
            # Check that the file exists
            self.assertTrue(os.path.exists(model_path))
            
            # Load the model
            loaded_model = MultimodalFusionModel.load(model_path)
            
            # Check that the loaded model has the same parameters
            self.assertEqual(loaded_model.fusion_dim, self.model.fusion_dim)
            self.assertEqual(loaded_model.output_dim, self.model.output_dim)


class TestModelOptimization(unittest.TestCase):
    """Test model optimization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal fusion model for testing
        self.model = MultimodalFusionModel(
            visual_embedding_dim=768,
            text_embedding_dim=768,
            fusion_dim=512,
            output_dim=1,
            device="cpu"
        )
        
        # Skip actual ONNX export in tests
        self.patcher = patch("torch.onnx.export")
        self.mock_onnx_export = self.patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_model_quantization(self):
        """Test model quantization."""
        quantizer = ModelQuantizer()
        
        # Test dynamic quantization (INT8)
        quantized_model = quantizer.quantize_dynamic(self.model)
        
        # The quantized model should have _packed_params attributes
        self.assertTrue(hasattr(quantized_model, "_is_quantized") or 
                       any("_packed_params" in name for name, _ in quantized_model.named_modules()))
    
    def test_onnx_export(self):
        """Test ONNX export functionality."""
        exporter = ONNXExporter()
        
        # Create test inputs
        visual_features = torch.rand(1, 768)
        text_features = torch.rand(1, 768)
        
        # Create a temporary file for the ONNX model
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            # Export the model
            exporter.export_model(
                model=self.model,
                output_path=tmp.name,
                visual_features=visual_features,
                text_features=text_features
            )
            
            # Check that export was called
            self.mock_onnx_export.assert_called_once()


if __name__ == '__main__':
    unittest.main()