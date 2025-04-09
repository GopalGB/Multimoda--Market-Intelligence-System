# Cross-Modal Audience Intelligence Platform (CAIP)

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive multimodal AI system that leverages visual, textual, and structured data to identify **causal drivers** of viewer engagement. CAIP combines state-of-the-art deep learning models with causal inference techniques to provide actionable insights for content creators and marketers.

## 🔍 Overview

The Cross-Modal Audience Intelligence Platform (CAIP) is designed to:

1. **Analyze content** across multiple modalities (text, image, metadata)
2. **Identify causal factors** that drive audience engagement
3. **Predict counterfactual outcomes** for content modifications
4. **Recommend optimizations** to maximize viewer engagement
5. **Explain results** through transparent causal relationships

## 🏗️ Architecture

CAIP consists of several interconnected components:

```
┌────────────────┐    ┌─────────────────┐    ┌───────────────────┐
│ Data Pipeline  │───▶│  Model Pipeline  │───▶│  Causal Pipeline  │
└────────────────┘    └─────────────────┘    └───────────────────┘
        │                     │                        │
        │                     │                        │
        ▼                     ▼                        ▼
┌────────────────┐    ┌─────────────────┐    ┌───────────────────┐
│ Preprocessing  │    │ Fusion Models   │    │ Structural Models │
└────────────────┘    └─────────────────┘    └───────────────────┘
                                │
                                │
                                ▼
                      ┌─────────────────┐    ┌───────────────────┐
                      │   API Server    │◀───│     RAG System    │
                      └─────────────────┘    └───────────────────┘
```

### Key Components:

- **Data Layer**: Connectors to various data sources, preprocessing pipelines, and feature engineering
- **Model Layer**: Visual models (CLIP), text models (RoBERTa), and multimodal fusion models
- **Causal Layer**: Structural causal models, counterfactual analysis, and causal feature identification
- **RAG System**: Retrieval-augmented generation for content recommendations and explanations
- **Serving Layer**: REST API, model serving infrastructure, and containerized deployment

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- CUDA-compatible GPU (recommended for training)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cross_modal_audience_platform.git
   cd cross_modal_audience_platform
   ```

2. Set up the environment and install dependencies:
   ```bash
   ./run_local.sh setup
   ```

3. Run tests to verify your installation:
   ```bash
   ./run_local.sh test
   ```

## 🛠️ Development Workflow

CAIP uses a modular architecture with well-defined interfaces between components. The development workflow is streamlined using the `run_local.sh` script:

```bash
# Run all unit tests
./run_local.sh test

# Run integration tests
./run_local.sh integration

# Run causal tests with specific configuration
./run_local.sh causal --config complex

# Collect training data
./run_local.sh collect --limit 1000

# Fine-tune the model
./run_local.sh finetune --epochs 10

# Start API server locally
./run_local.sh api

# Start API using Docker
./run_local.sh docker
```

For more options, run:
```bash
./run_local.sh help
```

## 📊 Causal Analysis

CAIP employs a structured approach to causal analysis:

1. **Feature Identification**: Extract relevant features from visual and textual content
2. **Causal Discovery**: Learn the causal graph structure from observational data
3. **Effect Estimation**: Quantify the causal effect of content features on engagement
4. **Counterfactual Analysis**: Predict engagement under hypothetical content modifications
5. **Recommendation Generation**: Suggest content changes to optimize engagement

## 🔄 Fine-tuning Pipeline

To fine-tune models on your own data:

1. Collect data:
   ```bash
   ./run_local.sh collect --source your_data_source --limit 5000
   ```

2. Fine-tune the model:
   ```bash
   ./run_local.sh finetune --config configs/fine_tuning.yaml
   ```

3. Evaluate the fine-tuned model:
   ```bash
   python scripts/evaluate_model.py --model-path models/saved/fusion_model_best.pt
   ```

## 🖥️ API Usage

Once the API server is running, you can interact with it using HTTP requests:

```python
import requests
import json

# Analyze content
response = requests.post(
    "http://localhost:5000/api/v1/analyze",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "text": "Exciting new product launch!",
        "image_url": "https://example.com/product_image.jpg",
        "metadata": {"category": "technology", "target_audience": "professionals"}
    }
)

# Parse results
results = response.json()
print(json.dumps(results, indent=2))
```

## 📁 Project Structure

```
.
├── causal/                  # Causal inference modules
├── data/                    # Data processing pipeline
│   ├── connectors/          # Data source connectors
│   └── preprocessing/       # Data preprocessing modules
├── evaluation/              # Evaluation metrics and benchmarks
├── models/                  # Machine learning models
│   ├── fusion/              # Multimodal fusion models
│   ├── text/                # Text models (RoBERTa)
│   └── visual/              # Visual models (CLIP)
├── notebooks/               # Jupyter notebooks for analysis
├── rag/                     # Retrieval-augmented generation
├── serving/                 # API and deployment components
│   └── kubernetes/          # Kubernetes deployment configs
├── tests/                   # Test suites
│   └── integration/         # Integration tests
├── scripts/                 # Utility scripts
├── configs/                 # Configuration files
├── requirements.txt         # Python dependencies
├── run_local.sh             # Local development script
└── README.md                # This file
```

## 🧪 Testing

CAIP includes comprehensive test suites:

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Causal Tests**: Validate causal discovery and inference
- **Benchmark Tests**: Measure performance and scalability

Run tests using:
```bash
./run_local.sh test         # Unit tests
./run_local.sh integration  # Integration tests
./run_local.sh causal       # Causal tests
```

## 🚢 Deployment

### Docker Deployment

Deploy the API server using Docker:

```bash
./run_local.sh docker
```

### Kubernetes Deployment

For production deployment on Kubernetes:

```bash
kubectl apply -f serving/kubernetes/
```

## 📚 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by production systems used at Nielsen
- Built with state-of-the-art open-source libraries and models