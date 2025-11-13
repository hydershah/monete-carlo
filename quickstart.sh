#!/bin/bash

# Sports Prediction System - Quick Start Script
# This script sets up and runs your first prediction

set -e

echo "=============================================="
echo "  Sports Prediction System - Quick Start"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"

# Check Docker
echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Warning: Docker not found. You'll need to set up PostgreSQL and Redis manually.${NC}"
    DOCKER_AVAILABLE=false
else
    echo -e "${GREEN}✓${NC} Docker found"
    DOCKER_AVAILABLE=true
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓${NC} Dependencies installed"

# Set up environment file
if [ ! -f ".env" ]; then
    echo ""
    echo "Setting up environment file..."
    cp .env.example .env
    echo -e "${YELLOW}⚠${NC}  Please edit .env and add your API keys:"
    echo "    - OPENAI_API_KEY (required for GPT analysis)"
    echo "    - THEODDS_API_KEY (optional for betting odds)"
    echo ""
    read -p "Press Enter after you've added your OPENAI_API_KEY to .env..."
fi

# Start Docker services if available
if [ "$DOCKER_AVAILABLE" = true ]; then
    echo ""
    echo "Starting databases..."
    docker-compose up -d
    echo "Waiting for databases to be ready..."
    sleep 10
    echo -e "${GREEN}✓${NC} Databases started"
fi

# Initialize database
echo ""
echo "Initializing database..."
python -m src.cli init
echo -e "${GREEN}✓${NC} Database initialized"

# Fetch NBA data
echo ""
echo "Fetching NBA game data..."
python -m src.cli fetch --sport nba --date today --with-odds || {
    echo -e "${YELLOW}Note: Could not fetch odds (TheOddsAPI key may not be set)${NC}"
    python -m src.cli fetch --sport nba --date today
}
echo -e "${GREEN}✓${NC} Game data fetched"

# Make predictions
echo ""
echo "Making predictions..."
read -p "Use GPT analysis? (costs API tokens) [y/N]: " use_gpt

if [[ $use_gpt =~ ^[Yy]$ ]]; then
    python -m src.cli predict --sport nba
else
    python -m src.cli predict --sport nba --no-gpt
fi

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "=============================================="
echo "  Next Steps"
echo "=============================================="
echo ""
echo "View predictions:"
echo "  python -m src.cli results --sport nba --days 1"
echo ""
echo "List games:"
echo "  python -m src.cli games --sport nba --date today"
echo ""
echo "Show statistics:"
echo "  python -m src.cli stats"
echo ""
echo "For more commands, run:"
echo "  python -m src.cli --help"
echo ""
echo "See GETTING_STARTED.md for detailed documentation"
echo ""
