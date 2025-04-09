# Changelog

All notable changes to the Cross-Modal Audience Intelligence Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2023-09-30

### Added

- Initial release of the Cross-Modal Audience Intelligence Platform
- Multimodal fusion model combining CLIP for visual features and RoBERTa for text features
- Cross-attention transformer architecture for feature fusion
- Structural causal models for identifying drivers of audience engagement
- Counterfactual analysis for predicting engagement impact of content changes
- RAG system for content recommendations and explanations
- RESTful API server with Docker containerization
- Kubernetes deployment configurations
- Fine-tuning pipeline for domain adaptation
- Comprehensive test suite including unit, integration, and causal tests
- Documentation and example notebooks

### Changed

- N/A (Initial release)

### Fixed

- N/A (Initial release)

## [0.9.0] - 2023-08-15

### Added

- Beta version with complete fusion model architecture
- Causal discovery and inference modules
- Initial API server implementation
- Docker containerization support
- Basic documentation

### Changed

- Improved performance of text feature extraction
- Enhanced visual feature extraction with domain adaptation
- Optimized fusion model inference speed

### Fixed

- Memory leak in the fusion model forward pass
- Race condition in the API server request handling
- Inconsistent results in counterfactual generation

## [0.8.0] - 2023-07-01

### Added

- Alpha version with preliminary fusion model
- Basic causal analysis capabilities
- Prototype API server
- Initial test suite

### Changed

- Refactored data preprocessing pipeline for better efficiency
- Improved model serialization and loading

### Fixed

- Data loading issues with certain image formats
- Incorrect text tokenization for special characters
- Model instability during training

## [0.7.0] - 2023-05-15

### Added

- Proof of concept implementation
- Initial data pipeline and connectors
- Baseline models for text and visual processing
- Experimental causal discovery

### Changed

- N/A (Initial feature implementation)

### Fixed

- N/A (Initial feature implementation) 