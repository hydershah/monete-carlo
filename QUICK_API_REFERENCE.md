# Quick API Reference

## Start the API

```bash
# First time setup
./quickstart.sh

# Start API server
./start_api.sh
```

API will be at: **http://localhost:8000**
Docs will be at: **http://localhost:8000/docs**

## Predict a Game

### Minimal Request (just team names)

```python
import requests

response = requests.post("http://localhost:8000/api/v1/predict", json={
    "home_team": "Lakers",
    "away_team": "Celtics",
    "sport": "nba"
})

print(response.json())
```

### With Team Statistics

```python
response = requests.post("http://localhost:8000/api/v1/predict", json={
    "home_team": "Los Angeles Lakers",
    "away_team": "Boston Celtics",
    "sport": "nba",

    # Team stats
    "home_ppg": 112.5,
    "away_ppg": 115.2,
    "home_defensive_rating": 108.3,
    "away_defensive_rating": 106.1,

    # Optional context
    "home_record": "10-5",
    "away_record": "12-3",
    "injuries": "Lakers: Anthony Davis (questionable - ankle)",
    "recent_form": "Celtics won last 4 games, Lakers 2-2",

    # Configuration
    "use_gpt": True,
    "monte_carlo_iterations": 10000
})

prediction = response.json()
```

### Response Example

```json
{
  "home_team": "Los Angeles Lakers",
  "away_team": "Boston Celtics",
  "sport": "nba",
  "home_win_probability": 0.45,
  "away_win_probability": 0.55,
  "predicted_home_score": 110.5,
  "predicted_away_score": 113.2,
  "confidence": 0.72,
  "recommendation": "LEAN AWAY - Boston Celtics (55.0%)",
  "models_used": ["monte_carlo", "gpt"],
  "gpt_analysis": "Celtics have strong momentum...",
  "gpt_adjustment": -0.03,
  "most_likely_score": "111-113",
  "processing_time_ms": 2341.5
}
```

## Using curl

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Lakers",
    "away_team": "Celtics",
    "sport": "nba",
    "use_gpt": true
  }'
```

## Batch Predictions

```python
response = requests.post("http://localhost:8000/api/v1/predict/batch", json={
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
})

batch = response.json()
for pred in batch['predictions']:
    print(f"{pred['away_team']} @ {pred['home_team']}: {pred['recommendation']}")
```

## GPT-Only Analysis (Fast & Cheap)

```python
response = requests.post("http://localhost:8000/api/v1/analyze", json={
    "home_team": "Cowboys",
    "away_team": "Eagles",
    "sport": "nfl",
    "injuries": "Cowboys: Dak Prescott (out)",
    "recent_form": "Eagles on 6-game win streak"
})
```

## Test the API

```bash
# Run test script
python test_api.py

# Or use interactive docs
open http://localhost:8000/docs
```

## Supported Sports

- `nba` - NBA Basketball
- `nfl` - NFL Football
- `mlb` - MLB Baseball
- `nhl` - NHL Hockey
- `soccer` - Soccer/Football

## Quick Tips

1. **Minimum required**: `home_team`, `away_team`, `sport`
2. **Better predictions**: Add `home_ppg`, `away_ppg`, defensive ratings
3. **Best predictions**: Include injuries, recent form, records
4. **Save money**: Set `use_gpt: false` for basic predictions
5. **Faster**: Use `/analyze` endpoint for GPT-only (no Monte Carlo)

## Common Patterns

### NFL Game

```python
{
    "home_team": "Dallas Cowboys",
    "away_team": "Philadelphia Eagles",
    "sport": "nfl",
    "home_record": "6-3",
    "away_record": "8-1",
    "injuries": "Cowboys: Dak Prescott (out - thumb)"
}
```

### Soccer Game

```python
{
    "home_team": "Manchester City",
    "away_team": "Liverpool",
    "sport": "soccer",
    "home_record": "18-2-3",
    "away_record": "16-4-3",
    "recent_form": "City won last 5, Liverpool drew last 2"
}
```

### MLB Game

```python
{
    "home_team": "New York Yankees",
    "away_team": "Boston Red Sox",
    "sport": "mlb",
    "recent_form": "Yankees ace pitcher on mound",
    "head_to_head": "Yankees won 4 of last 5 meetings"
}
```

## Error Handling

```python
try:
    response = requests.post("http://localhost:8000/api/v1/predict", json=data)
    response.raise_for_status()  # Raise exception for 4xx/5xx
    prediction = response.json()
except requests.exceptions.HTTPError as e:
    print(f"API Error: {e}")
    print(response.json())  # Error details
```

## Performance

- **GPT-only**: ~1-2 seconds
- **Monte Carlo only**: ~500ms
- **Full prediction**: ~2-3 seconds

## Costs (with GPT enabled)

- ~$0.001 per prediction
- ~$0.015 per day (15 NBA games)
- ~$0.45 per month (full NBA season)

Set `use_gpt: false` for free predictions!

## Full Documentation

- Complete API docs: [API_USAGE.md](API_USAGE.md)
- CLI usage: [GETTING_STARTED.md](GETTING_STARTED.md)
- Interactive docs: http://localhost:8000/docs
