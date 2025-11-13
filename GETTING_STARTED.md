# Getting Started with Sports Prediction System

This guide will help you set up and run the sports prediction system.

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (for databases)
- OpenAI API key (for GPT analysis)
- TheOddsAPI key (optional, for betting odds)

## Quick Start

### 1. Set Up Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# Required: OPENAI_API_KEY
# Optional: THEODDS_API_KEY
```

Example `.env`:
```
OPENAI_API_KEY=sk-your-key-here
THEODDS_API_KEY=your-theodds-key-here

DATABASE_URL=postgresql://sportsuser:sportspass123@localhost:5432/sports_predictions
REDIS_URL=redis://localhost:6379/0
```

### 3. Start Databases

```bash
# Start PostgreSQL and Redis using Docker
docker-compose up -d

# Wait a few seconds for databases to be ready
sleep 5

# Verify they're running
docker-compose ps
```

### 4. Initialize Database

```bash
# Create database tables
python -m src.cli init
```

### 5. Fetch Game Data

```bash
# Fetch NBA games for today
python -m src.cli fetch --sport nba --date today --with-odds

# Or fetch for a specific date
python -m src.cli fetch --sport nba --date 20241113
```

### 6. Make Predictions

```bash
# Predict today's NBA games with GPT analysis
python -m src.cli predict --sport nba

# Predict without GPT to save API costs
python -m src.cli predict --sport nba --no-gpt
```

### 7. View Results

```bash
# List today's games
python -m src.cli games --sport nba --date today

# View prediction results
python -m src.cli results --sport nba --days 7

# Show system statistics
python -m src.cli stats
```

## Detailed Workflow

### Daily Prediction Workflow

```bash
# 1. Fetch latest game data
python -m src.cli fetch --sport nba --date today --with-odds

# 2. Make predictions
python -m src.cli predict --sport nba

# 3. View predictions
python -m src.cli results --sport nba --days 1
```

### Multi-Sport Setup

```bash
# Fetch data for multiple sports
python -m src.cli fetch --sport nba --date today --with-odds
python -m src.cli fetch --sport nfl --date today --with-odds
python -m src.cli fetch --sport mlb --date today --with-odds

# Make predictions for each
python -m src.cli predict --sport nba
python -m src.cli predict --sport nfl
python -m src.cli predict --sport mlb
```

## Testing Individual Components

### Test ESPN API Client

```bash
python -m src.data.espn_client
```

### Test TheOddsAPI Client

```bash
# Requires THEODDS_API_KEY in .env
python -m src.data.theodds_client
```

### Test Monte Carlo Simulation

```bash
python -m src.simulations.monte_carlo
```

### Test Elo Model

```bash
python -m src.simulations.elo_model
```

### Test Poisson Model

```bash
python -m src.simulations.poisson_model
```

### Test GPT Analyzer

```bash
# Requires OPENAI_API_KEY in .env
python -m src.simulations.gpt_analyzer
```

## Understanding the Predictions

The system combines multiple models:

1. **Elo Ratings** (40% weight) - Team strength based on historical performance
2. **Monte Carlo Simulation** (40% weight) - 10,000+ simulations of the game
3. **GPT Qualitative Analysis** (20% adjustment) - Contextual factors like injuries, momentum

### Output Explained

```
Boston Celtics @ Los Angeles Lakers
  Time: 2024-11-13T19:00:00
  Prediction: 58.3% home / 41.7% away
  Confidence: 72.4%
  Expected Score: 112.5 - 108.2
  GPT Adjustment: -3.2%
  Recommendation: LEAN HOME - Los Angeles Lakers (58.3%)
```

- **Prediction**: Probability of each team winning
- **Confidence**: How certain the model is (based on model agreement)
- **Expected Score**: Predicted final score
- **GPT Adjustment**: How much GPT analysis shifted the probability
- **Recommendation**: Betting suggestion (LEAN HOME/AWAY or PASS)

## Cost Management

### Minimizing API Costs

1. **Use GPT sparingly**: Add `--no-gpt` flag to predictions
2. **Batch predictions**: Predict all games at once rather than individually
3. **Use GPT-4o-mini**: Already configured (cheapest option)
4. **Free data sources**: ESPN API is free (TheOddsAPI has free tier)

### Estimated Costs

**Daily NBA season (with GPT):**
- ~15 games/day × $0.001/game = ~$0.015/day
- ~$0.45/month for regular season

**Daily NBA season (without GPT):**
- $0/day (ESPN API is free)

**TheOddsAPI:**
- Free tier: 500 requests/month
- Paid: $49-$99/month for higher limits

## Troubleshooting

### Database Connection Error

```bash
# Check if databases are running
docker-compose ps

# Restart databases
docker-compose restart

# Check logs
docker-compose logs postgres
docker-compose logs redis
```

### API Key Errors

```bash
# Verify API keys are set
grep OPENAI_API_KEY .env
grep THEODDS_API_KEY .env

# Test OpenAI connection
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

### No Games Found

```bash
# Check if data was fetched
python -m src.cli stats

# Fetch data again
python -m src.cli fetch --sport nba --date today

# List games in database
python -m src.cli games --sport nba --date today
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Or create fresh virtual environment
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Advanced Usage

### Python API

You can also use the system programmatically:

```python
from src.models import get_db_context, Game
from src.simulations.hybrid_predictor import HybridPredictor

# Create predictor
predictor = HybridPredictor(use_gpt=True)

# Get game from database
with get_db_context() as db:
    game = db.query(Game).filter(Game.id == 1).first()

    # Make prediction
    prediction = predictor.predict_game(game, db, save_to_db=True)

    print(prediction)
```

### Backtesting

Coming in Phase 2 - Historical validation framework

### Web API

Coming in Phase 2 - FastAPI REST endpoints

## Next Steps

1. **Phase 2** (Weeks 3-4): REST API, enhanced caching, backtesting
2. **Phase 3** (Weeks 5-6): Multi-sport expansion, advanced models
3. **Phase 4** (Weeks 7-8): Production deployment, monitoring, optimization

## Getting Help

- Check [README.md](README.md) for project overview
- Review individual module docstrings
- Check logs in `logs/` directory
- Enable debug logging: Set `LOG_LEVEL=DEBUG` in `.env`

## Best Practices

1. **Fetch data daily**: Run fetch command each morning
2. **Review predictions**: Don't blindly follow recommendations
3. **Track performance**: Monitor accuracy over time
4. **Start small**: Test with paper bets before real money
5. **Understand the models**: Read the research documentation

## Disclaimer

This tool is for educational and research purposes. Sports betting involves risk. Always gamble responsibly and within your means.
