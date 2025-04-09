#!/usr/bin/env python
"""
Simple test script to verify model loading works.
"""
import torch
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import model classes
from models.text.roberta_model import RoBERTaWrapper
from models.visual.clip_model import CLIPWrapper
from models.fusion.fusion_model import MultimodalFusionModel, FusionTypes

def main():
    """Test model loading."""
    print("Testing model initialization...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize text model (use smaller model for testing)
    print("Initializing text model...")
    text_model = RoBERTaWrapper(
        model_name="prajjwal1/bert-tiny",
        device=device
    )
    print(f"Text model initialized with dimension: {text_model.get_embedding_dimension()}")
    
    # Initialize visual model
    print("Initializing visual model...")
    visual_model = CLIPWrapper(
        model_name="openai/clip-vit-base-patch32",
        device=device
    )
    print(f"Visual model initialized with dimension: {visual_model.get_embedding_dimension()}")
    
    # Initialize fusion model
    print("Initializing fusion model...")
    fusion_model = MultimodalFusionModel(
        visual_dim=visual_model.get_embedding_dimension(),
        text_dim=text_model.get_embedding_dimension(),
        fusion_dim=256,
        num_layers=2,
        fusion_type=FusionTypes.CROSS_ATTENTION,
        device=device
    )
    print("Fusion model initialized")
    
    # Test saving model
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "test_fusion_model.pt"
    print(f"Saving fusion model to {model_path}")
    fusion_model.save(str(model_path))
    
    # Test loading model
    print(f"Loading fusion model from {model_path}")
    loaded_model = MultimodalFusionModel.load(str(model_path), device=device)
    print("Model loaded successfully")
    
    print("All tests passed!")
    
if __name__ == "__main__":
    main() 