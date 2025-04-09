# Audience Intelligence API

The REST API backend for the Cross-Modal Audience Intelligence Platform (CAIP).

## Overview

This API provides endpoints for:

- Content analysis (text and image)
- Causal factor identification
- Counterfactual prediction
- Content optimization suggestions
- User authentication and API key management

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (optional)

### Installation

#### Local Development

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run the application:
   ```bash
   python app.py
   ```

#### Docker Deployment

1. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

## API Endpoints

### Authentication

- `POST /auth/register` - Register a new user
- `POST /auth/login` - Login and get access token
- `POST /auth/api-keys` - Create a new API key
- `GET /auth/api-keys` - List all API keys

### Content Analysis

- `POST /api/v1/analyze` - Analyze content (text and image)
- `POST /api/v1/causal-analysis` - Get detailed causal analysis
- `POST /api/v1/counterfactual` - Generate counterfactual predictions
- `POST /api/v1/optimize` - Get content optimization suggestions

### Status

- `GET /` - API health check
- `GET /api/v1/status` - Detailed API status with model information

## Request Examples

### Analyze Content

```json
POST /api/v1/analyze
{
  "url": "https://example.com/article",
  "title": "Example Article Title",
  "text": "This is the content of the article...",
  "primary_image": "https://example.com/image.jpg"
}
```

### Generate Counterfactual

```json
POST /api/v1/counterfactual
{
  "content_id": "550e8400-e29b-41d4-a716-446655440000",
  "factor_name": "Content Length",
  "factor_value": 0.75
}
```

## Development

### Project Structure

```
serving/
├── api/                    # API endpoints
│   ├── __init__.py
│   ├── auth.py             # Authentication endpoints
│   └── routes.py           # Main API routes
├── models/                 # Model handling
│   ├── __init__.py
│   └── model_manager.py    # Model loading and inference
├── utils/                  # Utility functions
│   └── __init__.py
├── app.py                  # Main application entry point
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker configuration
└── docker-compose.yml      # Docker Compose configuration
```

### Adding New Endpoints

1. Add new route functions in `api/routes.py`
2. Register routes with the blueprint
3. Add request validation with Pydantic models

## Testing

Run the tests with:

```bash
pytest
```

## Deployment

For production deployment, consider:

- Using a reverse proxy (Nginx)
- Setting up SSL/TLS
- Implementing proper database storage
- Adding monitoring and logging
- Scaling with Kubernetes 