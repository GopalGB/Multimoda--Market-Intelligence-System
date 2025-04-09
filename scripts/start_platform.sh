#!/bin/bash

# Cross-Modal Audience Intelligence Platform (CAIP) Startup Script
# This script sets up and starts all components of the CAIP platform for local development

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "========================================================"
echo "  Cross-Modal Audience Intelligence Platform (CAIP)     "
echo "========================================================"
echo -e "${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check if .env file exists, create if not
ENV_FILE="$PROJECT_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Creating default .env file...${NC}"
    cat > "$ENV_FILE" << EOF
# CAIP Environment Configuration
API_PORT=5000
DEBUG=True
MODEL_CACHE_DIR=./cache
LOG_LEVEL=INFO
SECRET_KEY=$(openssl rand -hex 32)
EOF
    echo -e "${GREEN}Created default .env file at $ENV_FILE${NC}"
else
    echo -e "${GREEN}Using existing .env file${NC}"
fi

# Load environment variables
set -a
source "$ENV_FILE"
set +a

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo -e "${GREEN}Using Python $PYTHON_VERSION${NC}"
    
    if [[ $(echo "$PYTHON_VERSION < 3.8" | bc) -eq 1 ]]; then
        echo -e "${RED}Error: Python 3.8 or higher is required.${NC}"
        exit 1
    fi
    PYTHON=python3
else
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    exit 1
fi

# Check if virtual environment exists, create if not
VENV_DIR="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON -m venv "$VENV_DIR"
    echo -e "${GREEN}Created virtual environment at $VENV_DIR${NC}"
    
    echo -e "${YELLOW}Installing dependencies...${NC}"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r "$PROJECT_ROOT/requirements.txt"
    echo -e "${GREEN}Dependencies installed${NC}"
else
    echo -e "${GREEN}Using existing virtual environment${NC}"
    source "$VENV_DIR/bin/activate"
fi

# Create cache directories if they don't exist
CACHE_DIR="$PROJECT_ROOT/cache"
MODELS_DIR="$PROJECT_ROOT/models/saved"

mkdir -p "$CACHE_DIR/text"
mkdir -p "$CACHE_DIR/visual"
mkdir -p "$MODELS_DIR"

echo -e "${GREEN}Cache directories created/verified${NC}"

# Check if models need to be downloaded
if [ ! -f "$MODELS_DIR/fusion_model.pt" ]; then
    echo -e "${YELLOW}Pre-trained fusion model not found. Checking if test model exists...${NC}"
    
    # Check if test model exists from integration tests
    TEST_MODEL="$PROJECT_ROOT/tests/test_data/models/fusion_model.pt"
    if [ -f "$TEST_MODEL" ]; then
        echo -e "${YELLOW}Using test fusion model for demonstration${NC}"
        cp "$TEST_MODEL" "$MODELS_DIR/fusion_model.pt"
    else
        echo -e "${YELLOW}No fusion model found. The system will initialize a new one on first run.${NC}"
    fi
fi

# Set mode based on argument
MODE=${1:-"api"}

case "$MODE" in
    "api")
        echo -e "${BLUE}Starting API server on port $API_PORT...${NC}"
        cd "$PROJECT_ROOT"
        
        # Check if running with Docker
        if command_exists docker && [ -f "$PROJECT_ROOT/serving/docker-compose.yml" ]; then
            echo -e "${YELLOW}Docker detected. Do you want to run using Docker? (y/n)${NC}"
            read -r USE_DOCKER
            
            if [[ "$USE_DOCKER" =~ ^[Yy]$ ]]; then
                cd "$PROJECT_ROOT/serving"
                echo -e "${BLUE}Starting API server with Docker...${NC}"
                docker-compose up --build
                exit 0
            fi
        fi
        
        # Start API server locally
        cd "$PROJECT_ROOT"
        echo -e "${BLUE}Starting API server locally...${NC}"
        export PYTHONPATH="$PROJECT_ROOT"
        python -m serving.app
        ;;
        
    "notebook")
        echo -e "${BLUE}Starting Jupyter notebook server...${NC}"
        cd "$PROJECT_ROOT"
        export PYTHONPATH="$PROJECT_ROOT"
        jupyter notebook --notebook-dir="$PROJECT_ROOT/notebooks"
        ;;
        
    "test")
        echo -e "${BLUE}Running integration tests...${NC}"
        cd "$PROJECT_ROOT"
        export PYTHONPATH="$PROJECT_ROOT"
        python -m unittest discover -s tests/integration
        ;;
        
    "help")
        echo -e "${BLUE}Usage:${NC}"
        echo "  ./start_platform.sh [mode]"
        echo ""
        echo "Modes:"
        echo "  api       Start the API server (default)"
        echo "  notebook  Start the Jupyter notebook server"
        echo "  test      Run integration tests"
        echo "  help      Show this help message"
        ;;
        
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Run './start_platform.sh help' for usage information"
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}" 