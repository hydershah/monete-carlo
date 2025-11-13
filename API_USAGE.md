# Sports Prediction API - Usage Guide

This guide shows how to use the Sports Prediction API to analyze and predict game outcomes.

## Starting the API

### Method 1: Using uvicorn directly
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Method 2: Using Python
```bash
python -m src.api.main
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs (interactive Swagger UI)
- **ReDoc**: http://localhost:8000/redoc (alternative documentation)

## API Endpoints

### 1. Predict Single Game

**POST** `/api/v1/predict`

Send game data and get back a prediction with:
- Win probabilities
- Expected final score
- Confidence level
- Betting recommendation
- GPT analysis (if enabled)
- Monte Carlo simulation results

**Example Request (Python):**
```python
import requests

game_data = {
    "home_team": "Los Angeles Lakers",
    "away_team": "Boston Celtics",
    "sport": "nba",
    "game_date": "2024-11-13",

    # Optional team statistics
    "home_ppg": 112.5,
    "away_ppg": 115.2,
    "home_defensive_rating": 108.3,
    "away_defensive_rating": 106.1,

    # Optional context
    "home_record": "10-5",
    "away_record": "12-3",
    "injuries": "Lakers: Anthony Davis (questionable - ankle)",
    "recent_form": "Celtics won last 4 games",

    # Configuration
    "use_gpt": True,
    "monte_carlo_iterations": 10000
}

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json=game_data
)

prediction = response.json()
print(f"Home win: {prediction['home_win_probability']:.1%}")
print(f"Recommendation: {prediction['recommendation']}")
```

**Example Request (curl):**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Lakers",
    "away_team": "Celtics",
    "sport": "nba",
    "home_ppg": 112.5,
    "away_ppg": 115.2,
    "use_gpt": true
  }'
```

**Response:**
```json
{
  "home_team": "Los Angeles Lakers",
  "away_team": "Boston Celtics",
  "sport": "nba",
  "game_date": "2024-11-13",
  "home_win_probability": 0.45,
  "away_win_probability": 0.55,
  "predicted_home_score": 110.5,
  "predicted_away_score": 113.2,
  "confidence": 0.72,
  "recommendation": "LEAN AWAY - Boston Celtics (55.0%)",
  "models_used": ["monte_carlo", "gpt"],
  "gpt_analysis": "Celtics have momentum with 4-game win streak...",
  "gpt_adjustment": -0.03,
  "most_likely_score": "111-113",
  "prediction_timestamp": "2024-11-13T10:30:00",
  "processing_time_ms": 2341.5
}
```

### 2. Batch Predictions

**POST** `/api/v1/predict/batch`

Predict multiple games at once.

**Example:**
```python
batch_data = {
    "games": [
        {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "sport": "nba",
            "home_ppg": 112.5,
            "away_ppg": 115.2
        },
        {
            "home_team": "Warriors",
            "away_team": "Nets",
            "sport": "nba",
            "home_ppg": 118.3,
            "away_ppg": 110.1
        }
    ]
}

response = requests.post(
    "http://localhost:8000/api/v1/predict/batch",
    json=batch_data
)
```

### 3. GPT-Only Analysis (Fast & Cheap)

**POST** `/api/v1/analyze`

Get only GPT analysis without Monte Carlo simulation. Faster and uses fewer API tokens.

**Example:**
```python
game_data = {
    "home_team": "Dallas Cowboys",
    "away_team": "Philadelphia Eagles",
    "sport": "nfl",
    "injuries": "Cowboys: Dak Prescott (out)",
    "recent_form": "Eagles on 6-game win streak"
}

response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json=game_data
)
```

### 4. Health Check

**GET** `/health`

Check API status and configuration.

```bash
curl http://localhost:8000/health
```

### 5. Supported Sports

**GET** `/api/v1/sports`

Get list of all supported sports.

```bash
curl http://localhost:8000/api/v1/sports
```

## Request Parameters

### Required Fields
- `home_team` (string): Home team name
- `away_team` (string): Away team name
- `sport` (string): Sport ID (nba, nfl, mlb, nhl, soccer)

### Optional Statistics
- `home_ppg` (float): Home team points/goals per game
- `away_ppg` (float): Away team points/goals per game
- `home_offensive_rating` (float): Offensive rating
- `away_offensive_rating` (float): Offensive rating
- `home_defensive_rating` (float): Defensive rating
- `away_defensive_rating` (float): Defensive rating

### Optional Context
- `game_date` (string): Game date (YYYY-MM-DD)
- `home_record` (string): Home team record (e.g., "10-5")
- `away_record` (string): Away team record
- `injuries` (string): Injury report
- `recent_form` (string): Recent performance
- `head_to_head` (string): Historical matchup info
- `betting_odds` (object): Current odds if available

### Configuration
- `use_gpt` (boolean): Enable GPT analysis (default: true)
- `use_monte_carlo` (boolean): Enable Monte Carlo (default: true)
- `monte_carlo_iterations` (int): Number of simulations (default: 10000)

## Response Fields

- `home_win_probability`: Probability home team wins (0-1)
- `away_win_probability`: Probability away team wins (0-1)
- `draw_probability`: Probability of draw (if applicable)
- `predicted_home_score`: Expected home team score
- `predicted_away_score`: Expected away team score
- `confidence`: Prediction confidence level (0-1)
- `recommendation`: Betting recommendation
- `models_used`: List of models used
- `gpt_analysis`: GPT's qualitative analysis (if enabled)
- `gpt_adjustment`: How much GPT adjusted the prediction
- `most_likely_score`: Most probable final score
- `processing_time_ms`: Time taken to generate prediction

## Code Examples

### Python with requests

```python
import requests

API_URL = "http://localhost:8000"

def predict_game(home_team, away_team, sport, **kwargs):
    """Helper function to predict a game."""
    data = {
        "home_team": home_team,
        "away_team": away_team,
        "sport": sport,
        **kwargs
    }

    response = requests.post(f"{API_URL}/api/v1/predict", json=data)
    return response.json()

# Example usage
prediction = predict_game(
    home_team="Lakers",
    away_team="Celtics",
    sport="nba",
    home_ppg=112.5,
    away_ppg=115.2,
    use_gpt=True
)

print(f"Prediction: {prediction['recommendation']}")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function predictGame(gameData) {
  const response = await axios.post(
    'http://localhost:8000/api/v1/predict',
    gameData
  );
  return response.data;
}

// Example
const prediction = await predictGame({
  home_team: "Lakers",
  away_team: "Celtics",
  sport: "nba",
  home_ppg: 112.5,
  away_ppg: 115.2,
  use_gpt: true
});

console.log(`Home win: ${prediction.home_win_probability}`);
```

### cURL

```bash
# Simple prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Lakers",
    "away_team": "Celtics",
    "sport": "nba",
    "use_gpt": true
  }' | jq .

# With full context
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Lakers",
    "away_team": "Celtics",
    "sport": "nba",
    "home_ppg": 112.5,
    "away_ppg": 115.2,
    "home_defensive_rating": 108.3,
    "away_defensive_rating": 106.1,
    "injuries": "Lakers: AD questionable",
    "use_gpt": true,
    "monte_carlo_iterations": 10000
  }' | jq .
```

## Testing the API

Run the test script:
```bash
python test_api.py
```

Or use the interactive Swagger docs:
```
http://localhost:8000/docs
```

## Performance & Costs

### Processing Time
- **GPT-only analysis**: ~1-2 seconds
- **Monte Carlo (10K iterations)**: ~500ms
- **Full hybrid prediction**: ~2-3 seconds

### API Costs (with GPT enabled)
- ~$0.001 per prediction
- ~$0.015 per day (15 NBA games)
- ~$0.45 per month (full NBA season)

### Optimization Tips
1. **Disable GPT for batch predictions** to save cost
2. **Reduce monte_carlo_iterations** (5000 is often sufficient)
3. **Use /analyze endpoint** for quick insights
4. **Cache predictions** to avoid duplicate requests

## Error Handling

The API returns standard HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid input)
- `500`: Server error (prediction failed)

Error response format:
```json
{
  "error": "Prediction failed",
  "detail": "Missing required field: home_team",
  "timestamp": "2024-11-13T10:30:00"
}
```

## Rate Limiting

Currently no rate limiting is implemented. For production use, consider:
- Adding API key authentication
- Implementing rate limits
- Caching predictions with Redis

## Next Steps

1. **Deploy to production**: Use Gunicorn or similar
2. **Add authentication**: API keys or JWT
3. **Add caching**: Redis for frequently requested games
4. **Add webhooks**: Real-time updates
5. **Add database integration**: Store predictions

## Support

- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Supported sports: http://localhost:8000/api/v1/sports
