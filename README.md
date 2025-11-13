# Sports Prediction Model

A comprehensive sports prediction system using Monte Carlo simulations and GPT analysis for multiple sports leagues.

## 🚀 Quick Start

### Option 1: Run as API Service (Recommended)

```bash
# Setup (one time)
./quickstart.sh

# Start the API
./start_api.sh
```

The API will be available at **http://localhost:8000** with interactive docs at **/docs**

**Send game data to get predictions:**

```python
import requests

prediction = requests.post("http://localhost:8000/api/v1/predict", json={
    "home_team": "Los Angeles Lakers",
    "away_team": "Boston Celtics",
    "sport": "nba",
    "home_ppg": 112.5,
    "away_ppg": 115.2,
    "injuries": "Lakers: AD questionable",
    "use_gpt": True
}).json()

print(f"Prediction: {prediction['recommendation']}")
print(f"Home Win: {prediction['home_win_probability']:.1%}")
```

### Option 2: Use CLI

```bash
./quickstart.sh
python -m src.cli predict --sport nba
```

See [API_USAGE.md](API_USAGE.md) for complete API documentation or [GETTING_STARTED.md](GETTING_STARTED.md) for CLI usage.

## Features

- **Multi-Sport Support**: NBA, NFL, MLB, NHL, and Soccer
- **Hybrid Predictions**: Combines Monte Carlo simulations (10K+ iterations) with GPT qualitative analysis
- **Multiple Models**: Elo ratings, Poisson distribution, Monte Carlo, and GPT-4o Mini
- **CLI Interface**: Easy-to-use command-line tools for predictions and analysis
- **Real-time Data**: Integration with ESPN (free) and TheOddsAPI (optional)
- **Database Storage**: PostgreSQL + TimescaleDB for efficient time-series data
- **Caching**: Redis for optimal performance
- **Production Ready**: Docker support, comprehensive logging, and monitoring

## Tech Stack

- **Backend**: Python 3.11+, FastAPI
- **Database**: PostgreSQL + TimescaleDB
- **Cache**: Redis
- **ML/Stats**: NumPy, SciPy, Pandas, Scikit-learn
- **LLM**: OpenAI GPT-4o/GPT-4o Mini
- **Data APIs**: ESPN (unofficial) + TheOddsAPI

## Project Structure

```
.
├── src/
│   ├── api/              # FastAPI routes and endpoints
│   ├── data/             # Data fetching and processing
│   ├── models/           # Database models and schemas
│   ├── simulations/      # Monte Carlo and statistical models
│   └── utils/            # Helper functions and utilities
├── config/               # Configuration files
├── tests/                # Test suite
├── data/                 # Data storage
│   ├── raw/             # Raw data from APIs
│   └── processed/       # Processed data
└── logs/                 # Application logs
```

## How It Works

The system combines multiple prediction models:

1. **Elo Rating System** (40% weight)
   - Tracks team strength over time
   - Updates after each game
   - Sport-specific configurations

2. **Monte Carlo Simulation** (40% weight)
   - Runs 10,000+ simulations per game
   - Uses Poisson distribution for low-scoring sports
   - Normal distribution for high-scoring sports
   - Generates probability distributions

3. **GPT-4o Mini Analysis** (20% adjustment)
   - Analyzes qualitative factors (injuries, momentum, etc.)
   - Identifies situational advantages
   - Adjusts probabilities based on context
   - Cost-optimized with token management

4. **Hybrid Combination**
   - Weighted average of all models
   - Bayesian adjustment from GPT insights
   - Confidence scoring based on model agreement

## CLI Commands

```bash
# Initialize database
python -m src.cli init

# Fetch game data
python -m src.cli fetch --sport nba --date today --with-odds

# Make predictions
python -m src.cli predict --sport nba

# View results
python -m src.cli results --sport nba --days 7

# List games
python -m src.cli games --sport nba --date today

# Show statistics
python -m src.cli stats
```

## Installation

1. **Quick Setup**
```bash
./quickstart.sh
```

2. **Manual Setup**

Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. **Set up databases**
```bash
# PostgreSQL (via Docker)
docker run --name sports-postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres:16

# Redis (via Docker)
docker run --name sports-redis -p 6379:6379 -d redis:7
```

6. **Run database migrations**
```bash
# Coming soon
```

## Usage

### CLI Interface

```bash
# Get prediction for a specific game
python -m src.cli predict --game-id 12345

# Backtest models on historical data
python -m src.cli backtest --sport nba --start-date 2024-01-01

# Fetch and store game data
python -m src.cli fetch --sport nba --date 2024-11-13
```

### REST API

```bash
# Start the API server
uvicorn src.api.main:app --reload

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### Example API Calls

```bash
# Get predictions for today's games
curl http://localhost:8000/api/v1/predictions?sport=nba&date=today

# Get specific game prediction
curl http://localhost:8000/api/v1/predictions/game/12345

# Get model performance metrics
curl http://localhost:8000/api/v1/metrics?sport=nba
```

## Current Status

### ✅ Phase 1 Complete - MVP Foundation

- ✅ Project structure and configuration
- ✅ ESPN API client (game data, free)
- ✅ TheOddsAPI client (betting odds, optional)
- ✅ PostgreSQL + TimescaleDB database schema
- ✅ NBA data ingestion pipeline
- ✅ Poisson distribution model
- ✅ Elo rating system
- ✅ Monte Carlo simulation engine (10K iterations)
- ✅ GPT-4o Mini integration
- ✅ Hybrid prediction system
- ✅ CLI interface
- ✅ Docker Compose setup
- ✅ **FastAPI REST API with prediction endpoints**
- ✅ **Standalone prediction service (no database required)**
- ✅ Comprehensive documentation

### 🔄 Next: Phase 2 - API & Infrastructure (Weeks 3-4)

- ✅ FastAPI REST endpoints
- [ ] Enhanced Redis caching
- [ ] Automated daily jobs (Celery/Airflow)
- [ ] Backtesting framework
- [ ] Performance tracking dashboard
- [ ] API authentication

### 📋 Future: Phase 3 - Multi-Sport Expansion (Weeks 5-6)

- [ ] NFL support with sport-specific models
- [ ] MLB integration
- [ ] NHL/Soccer support
- [ ] Cross-sport analytics
- [ ] Model ensemble optimization

### 🎯 Future: Phase 4 - Production Features (Weeks 7-8)

- [ ] Advanced monitoring (Prometheus + Grafana)
- [ ] Portfolio management
- [ ] Kelly Criterion bet sizing
- [ ] Comprehensive testing
- [ ] Production deployment guides

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Disclaimer

This tool is for educational and research purposes only. Sports betting involves risk. Always gamble responsibly and within your means.
