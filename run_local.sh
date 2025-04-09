#!/bin/bash
# Cross-Modal Audience Intelligence Platform (CAIP)
# Local Development and Testing Script

set -e

# Set environmental variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display header
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}    Cross-Modal Audience Intelligence Platform (CAIP)    ${NC}"
echo -e "${BLUE}                  Local Development Tool                 ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Function to check if Python 3.9+ is installed
check_python() {
    if command -v python3 >/dev/null 2>&1; then
        python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if (( $(echo "$python_version >= 3.9" | bc -l) )); then
            echo -e "${GREEN}✓ Python $python_version detected${NC}"
        else
            echo -e "${RED}✗ Python 3.9+ required, but $python_version found${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ Python 3 not found${NC}"
        exit 1
    fi
}

# Function to check Docker installation
check_docker() {
    if command -v docker >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Docker detected${NC}"
    else
        echo -e "${YELLOW}⚠ Docker not found. Some features may not work.${NC}"
    fi
}

# Function to install dependencies
install_dependencies() {
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Function to run tests
run_tests() {
    test_type=$1
    echo -e "${BLUE}Running $test_type tests...${NC}"
    
    case $test_type in
        "unit")
            pytest tests/test_*.py -v
            ;;
        "integration")
            pytest tests/integration/test_*.py -v
            ;;
        "causal")
            python scripts/run_causal_tests.py
            ;;
        "all")
            pytest
            python scripts/run_causal_tests.py
            ;;
        *)
            echo -e "${RED}Unknown test type: $test_type${NC}"
            echo "Available options: unit, integration, causal, all"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}✓ $test_type tests completed${NC}"
}

# Function to collect training data
collect_data() {
    echo -e "${BLUE}Collecting training data...${NC}"
    python scripts/collect_training_data.py --output_dir data/processed --num_samples 100
    echo -e "${GREEN}✓ Data collection completed${NC}"
}

# Function to run fine-tuning
run_fine_tuning() {
    echo -e "${BLUE}Running model fine-tuning...${NC}"
    python scripts/fine_tune.py --config_path configs/fine_tuning.yaml
    echo -e "${GREEN}✓ Fine-tuning completed${NC}"
}

# Function to start API server
start_api() {
    echo -e "${BLUE}Starting API server...${NC}"
    cd serving
    python app.py
}

# Function to start Docker services
start_docker() {
    echo -e "${BLUE}Starting Docker services...${NC}"
    cd serving
    docker-compose up --build
}

# Function to clean project
clean_project() {
    echo -e "${BLUE}Cleaning project...${NC}"
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type f -name "*.log" -delete
    echo -e "${GREEN}✓ Project cleaned${NC}"
}

# Function to check status and environment
check_status() {
    echo -e "${BLUE}Checking environment status...${NC}"
    
    # Check if virtual environment is active
    if [[ -n $VIRTUAL_ENV ]]; then
        echo -e "${GREEN}✓ Virtual environment is active: $VIRTUAL_ENV${NC}"
    else
        echo -e "${YELLOW}⚠ No virtual environment active${NC}"
    fi
    
    # Check Python and package versions
    echo -e "\nPython version:"
    python3 --version
    
    echo -e "\nKey packages:"
    pip freeze | grep -E "torch|transformers|pandas|numpy|pytest|fastapi|ray"
    
    # Check model files
    echo -e "\nChecking model files:"
    if [ -d "models/checkpoints" ]; then
        echo -e "${GREEN}✓ Model checkpoints found${NC}"
        ls -lh models/checkpoints
    else
        echo -e "${YELLOW}⚠ No model checkpoints directory found${NC}"
    fi
    
    # Check if API server is running
    if pgrep -f "python.*app.py" > /dev/null; then
        echo -e "\n${GREEN}✓ API server is running${NC}"
    else
        echo -e "\n${YELLOW}⚠ API server is not running${NC}"
    fi
    
    # Check Docker containers
    if command -v docker >/dev/null 2>&1; then
        echo -e "\nDocker containers:"
        docker ps --filter "name=caip-*"
    fi
}

# Function to display help
show_help() {
    echo -e "Usage: $0 [command]"
    echo -e "\nAvailable commands:"
    echo -e "  ${GREEN}setup${NC}        - Check environment and install dependencies"
    echo -e "  ${GREEN}test${NC}         - Run tests (args: unit, integration, causal, all)"
    echo -e "  ${GREEN}collect${NC}      - Collect training data"
    echo -e "  ${GREEN}finetune${NC}     - Run fine-tuning pipeline"
    echo -e "  ${GREEN}api${NC}          - Start the API server"
    echo -e "  ${GREEN}docker${NC}       - Start services with Docker"
    echo -e "  ${GREEN}clean${NC}        - Clean project files (__pycache__, .pyc, etc.)"
    echo -e "  ${GREEN}status${NC}       - Check environment status"
    echo -e "  ${GREEN}help${NC}         - Show this help message"
    echo -e "\nExamples:"
    echo -e "  $0 setup"
    echo -e "  $0 test unit"
    echo -e "  $0 api"
}

# Main command processing
case $1 in
    "setup")
        check_python
        check_docker
        install_dependencies
        ;;
    "test")
        test_type=${2:-"all"}
        run_tests $test_type
        ;;
    "collect")
        collect_data
        ;;
    "finetune")
        run_fine_tuning
        ;;
    "api")
        start_api
        ;;
    "docker")
        start_docker
        ;;
    "clean")
        clean_project
        ;;
    "status")
        check_status
        ;;
    "help")
        show_help
        ;;
    *)
        show_help
        ;;
esac

exit 0 