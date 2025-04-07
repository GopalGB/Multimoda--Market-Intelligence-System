# Cross-Modal Audience Intelligence Platform (CAIP)

A sophisticated multimodal AI system that fuses text, image, and structured data to derive actionable insights on audience engagement. The platform leverages state-of-the-art deep learning models, causal inference, and retrieval-augmented generation to provide accurate predictions and meaningful explanations.

## Key Features

- **Multimodal Fusion**: Combines CLIP (visual features) and RoBERTa (text features) via a custom cross-attention transformer for comprehensive content analysis
- **Causal Inference**: Identifies true drivers of audience engagement using structural causal models to differentiate causation from correlation
- **Retrieval-Augmented Generation**: Enhances content insights with a hybrid RAG system that combines dense and sparse retrieval
- **Production-Ready Serving**: Optimized inference through ONNX, TorchScript, and quantization techniques
- **Explainable AI**: Provides counterfactual explanations and feature attribution for interpretable predictions

## Architecture

The CAIP system consists of the following components:

- **Data Layer**: Connectors for streaming platforms, social media, and proprietary panel data
- **Preprocessing Layer**: Feature extraction, normalization, and integration across modalities
- **Model Layer**: Multimodal fusion models combining visual and textual features
- **Causal Layer**: Structural causal models for identifying causal relationships
- **RAG Layer**: Retrieval and generation components for context-aware insights
- **Serving Layer**: FastAPI and Ray Serve deployments for scalable inference

## Installation

```bash
# Clone the repository
git clone https://github.com/nielsen/audience-intelligence.git
cd audience-intelligence

# Install with basic dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"

# Install with GPU support
pip install -e ".[gpu]"

# Install with causal inference tools
pip install -e ".[causal]"
```

## Quick Start

### Prediction API

```bash
# Start the FastAPI server
python -m serving.api

# Or use Ray Serve for distributed deployment
python -m serving.ray_serve
```

### Python Client

```python
import requests
import json
import base64
from PIL import Image
import io

# Encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Predict engagement
def predict_engagement(text, image_path=None):
    url = "http://localhost:8000/api/v1/predict/engagement"
    
    payload = {
        "text_content": text,
    }
    
    if image_path:
        payload["visual_content_url"] = image_path
    
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
text = "New streaming series explores artificial intelligence in modern society"
image_path = "path/to/show_thumbnail.jpg"
result = predict_engagement(text, image_path)
print(f"Engagement prediction: {result['engagement_prediction']}")
```

## Documentation

For full documentation, visit [docs/](docs/) or see the specific component guides:

- [Data Connectors](docs/data-connectors.md)
- [Model Architecture](docs/model-architecture.md)
- [Causal Inference](docs/causal-inference.md)
- [RAG System](docs/rag-system.md)
- [Deployment Guide](docs/deployment.md)

## Notebooks

Explore our Jupyter notebooks for examples and analysis:

- [Data Exploration](notebooks/data_exploration.ipynb): Analyze audience data patterns
- [Model Training](notebooks/model_training.ipynb): Train and evaluate fusion models
- [Causal Analysis](notebooks/causal_analysis.ipynb): Discover causal relationships in audience behavior

## Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=.
```

## Citation

If you use CAIP in your research, please cite our paper:

```
@article{nielsen2023caip,
  title={Cross-Modal Audience Intelligence: Multimodal Modeling of Content Engagement},
  author={Nielsen AI Team},
  journal={Proceedings of the Conference on Applied AI},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.