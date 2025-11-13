#!/bin/bash

# Start the Sports Prediction API

echo "=================================================="
echo "  Starting Sports Prediction API"
echo "=================================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Copy .env.example to .env and add your API keys:"
    echo "  cp .env.example .env"
    echo ""
    read -p "Continue anyway? [y/N]: " continue
    if [[ ! $continue =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import fastapi" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
}

echo "✓ Dependencies ready"
echo ""

# Start the API
echo "Starting API server..."
echo "  - API: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - Health: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
