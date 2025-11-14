# Historical Data Ingestion Guide

## Overview

This guide explains how to fetch historical sports data from ESPN API to populate your backtesting system with real game data.

## Quick Start

```bash
# 1. Test the pipeline (fetches last 30 days as sample)
python scripts/test_data_pipeline.py

# 2. Ingest 3 seasons of NBA data (takes 10-15 minutes)
python scripts/ingest_historical_data.py

# 3. Use the data for backtesting
python -m src.simulations.backtesting
```

## Data Sources

### ESPN API (Primary)
- ✅ **Free** - No API key required
- ✅ **Comprehensive** - NBA, NFL, MLB, NHL
- ✅ **Historical** - Multiple seasons available
- ✅ **Reliable** - Used by ESPN.com and mobile apps
- ⚠️ **Unofficial** - No official documentation, may change
- ⚠️ **Rate Limited** - Requires 1-2 second delays between requests

### ESPN API Endpoints Used
```
Base URL: https://site.api.espn.com/apis/site/v2/sports

Scoreboard: /basketball/nba/scoreboard?dates=YYYYMMDD
Teams:      /basketball/nba/teams
Schedule:   /basketball/nba/scoreboard?dates=YYYYMMDD-YYYYMMDD
```

## Features

### 1. Historical Data Fetcher (`src/data/historical_data_fetcher.py`)

Fetches complete season data from ESPN API:

```python
from src.data.historical_data_fetcher import HistoricalDataFetcher

fetcher = HistoricalDataFetcher()

# Fetch 3 seasons of NBA data
games = fetcher.fetch_historical_games('nba', seasons=3)

# Save to database
fetcher.save_to_database(games)

# Save to CSV
fetcher.save_to_csv(games, 'data/nba_historical.csv')
```

**Supported Leagues:**
- `nba` - NBA (1,230 games/season)
- `nfl` - NFL (285 games/season)
- `mlb` - MLB (2,430 games/season)
- `nhl` - NHL (1,312 games/season)

### 2. Data Ingestion Script (`scripts/ingest_historical_data.py`)

Command-line tool for batch data ingestion:

```bash
# Basic usage - fetch 3 seasons of NBA data
python scripts/ingest_historical_data.py

# Fetch 5 seasons of NFL data
python scripts/ingest_historical_data.py --league nfl --seasons 5

# Fetch multiple leagues
python scripts/ingest_historical_data.py --league nba nfl mlb --seasons 2

# Save to CSV only (skip database)
python scripts/ingest_historical_data.py --csv-only

# Custom output directory
python scripts/ingest_historical_data.py --output-dir ./my_data

# Faster rate (risky - may get blocked)
python scripts/ingest_historical_data.py --rate-limit 0.5
```

### 3. Test Pipeline (`scripts/test_data_pipeline.py`)

Validates the entire pipeline:

```bash
python scripts/test_data_pipeline.py
```

Tests:
- ✅ ESPN Client connectivity
- ✅ Data fetching and parsing
- ✅ Data format compatibility with backtesting
- ✅ CSV save/load functionality
- ✅ Data validation

## Data Format

### HistoricalGame Object

```python
@dataclass
class HistoricalGame:
    game_id: str          # Unique ESPN game ID
    game_date: datetime   # Game date/time
    league: str           # NBA, NFL, etc.
    season: int           # Season year

    # Teams
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str

    # Scores
    home_score: int
    away_score: int

    # Betting lines (if available)
    spread: Optional[float]
    total: Optional[float]
    home_ml: Optional[float]
    away_ml: Optional[float]

    # Metadata
    status: str
    venue: Optional[str]
    neutral_site: bool
    completed: bool
    playoff: bool
```

### Database Schema

Data is saved to `games_history` table:

```sql
CREATE TABLE games_history (
    game_id VARCHAR PRIMARY KEY,
    game_date DATE NOT NULL,
    league VARCHAR NOT NULL,
    season INTEGER,

    home_team_id VARCHAR,
    home_team_name VARCHAR,
    away_team_id VARCHAR,
    away_team_name VARCHAR,

    home_score INTEGER,
    away_score INTEGER,
    home_win BOOLEAN,

    spread FLOAT,
    total FLOAT,
    home_covered BOOLEAN,
    went_over BOOLEAN,

    status VARCHAR,
    venue VARCHAR,
    neutral_site BOOLEAN
);
```

## Backtesting Integration

The fetched data is formatted to work directly with the backtesting framework:

```python
from src.data.historical_data_fetcher import HistoricalDataFetcher
from src.simulations.backtesting import WalkForwardAnalyzer
import pandas as pd

# 1. Load historical data
fetcher = HistoricalDataFetcher()
games = fetcher.load_from_csv('data/historical/nba_historical_3seasons.csv')
df = fetcher.get_games_dataframe(games)

# 2. Run backtesting
from sklearn.linear_model import LogisticRegression

analyzer = WalkForwardAnalyzer(
    train_window=500,
    test_window=100,
    step_size=50
)

model = LogisticRegression()
results = analyzer.analyze(model, df)

# 3. View results
from src.simulations.backtesting import create_backtest_report
print(create_backtest_report(results))
```

## Data Validation

The fetcher includes comprehensive validation:

```python
validation = fetcher.validate_data_quality(games)

# Outputs:
# {
#     'total_games': 3690,
#     'date_range': {'start': '2021-10-19', 'end': '2024-06-17', 'span_days': 973},
#     'seasons_covered': 3,
#     'teams_count': 30,
#     'completeness': {
#         'missing_scores': 0,
#         'missing_odds': {'spread': 245, 'total': 245},
#         'games_per_season': {'2021': 1230, '2022': 1230, '2023': 1230}
#     },
#     'quality_checks': {
#         'duplicate_games': 0,
#         'invalid_scores': 0,
#         'future_dates': 0
#     }
# }
```

## Performance & Rate Limiting

### ESPN API Rate Limits
- **No official limits** - But be respectful
- **Recommended:** 1-2 seconds between requests
- **Risk:** Faster rates may get IP temporarily blocked

### Ingestion Times (with 1s rate limit)

| League | Games/Season | Seasons | Time Estimate |
|--------|--------------|---------|---------------|
| NBA    | 1,230        | 3       | 10-15 min     |
| NFL    | 285          | 3       | 3-5 min       |
| MLB    | 2,430        | 3       | 20-25 min     |
| NHL    | 1,312        | 3       | 12-15 min     |

**Note:** Times include API requests + date range scanning (checks each day for games)

### Optimization Tips

1. **Parallel fetching** (multiple leagues at once):
```bash
# Fetch NBA and NFL simultaneously in different terminals
python scripts/ingest_historical_data.py --league nba &
python scripts/ingest_historical_data.py --league nfl &
```

2. **Resume from CSV**:
```python
# If ingestion fails, you can resume from CSV
fetcher = HistoricalDataFetcher()
games = fetcher.load_from_csv('data/historical/nba_historical_3seasons.csv')
fetcher.save_to_database(games)
```

3. **Use CSV-only mode for bulk fetching**:
```bash
# Fetch without database (faster)
python scripts/ingest_historical_data.py --csv-only

# Later, bulk insert to database
python scripts/bulk_insert_from_csv.py
```

## Troubleshooting

### Issue: "No games found"
**Cause:** League is in off-season or date range has no games
**Solution:** Try different date range or league

### Issue: "API request failed"
**Cause:** ESPN API is down or changed
**Solution:**
1. Check your internet connection
2. Try manual request: `curl "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"`
3. Increase rate limit: `--rate-limit 2.0`

### Issue: "Database connection failed"
**Cause:** PostgreSQL not running or wrong credentials
**Solution:**
1. Check database is running: `psql -h localhost -U user -d gamelens`
2. Use `--csv-only` to save data without database
3. Set correct DATABASE_URL environment variable

### Issue: "Duplicate games"
**Cause:** Running ingestion multiple times
**Solution:** This is OK - the script updates existing games

## Advanced Usage

### Custom Date Ranges

```python
from src.data.historical_data_fetcher import HistoricalDataFetcher
from datetime import datetime

fetcher = HistoricalDataFetcher()

# Fetch specific season
games = fetcher.fetch_historical_games(
    league='nba',
    seasons=1,
    start_year=2022  # 2022-2023 season
)
```

### Incremental Updates

```python
# Fetch only recent games (last 30 days)
from datetime import datetime, timedelta

start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()

games = []
current = start_date
while current <= end_date:
    date_str = current.strftime("%Y%m%d")
    scoreboard = fetcher.espn_client.get_scoreboard('nba', date=date_str)
    daily_games = fetcher.espn_client.parse_games(scoreboard)
    games.extend(daily_games)
    current += timedelta(days=1)
```

### Custom Filtering

```python
# Filter for specific teams
lakers_games = [
    g for g in games
    if 'Lakers' in g.home_team_name or 'Lakers' in g.away_team_name
]

# Filter for playoffs
playoff_games = [g for g in games if g.playoff]

# Filter for games with betting lines
games_with_odds = [g for g in games if g.spread is not None]
```

## Next Steps

1. **Test the pipeline**: `python scripts/test_data_pipeline.py`
2. **Fetch historical data**: `python scripts/ingest_historical_data.py`
3. **Validate data**: Check CSV files in `data/historical/`
4. **Run backtesting**: Use the fetched data with your models
5. **Automate updates**: Set up cron job to fetch daily results

## Resources

- [ESPN API Gist](https://gist.github.com/akeaswaran/b48b02f1c94f873c6655e7129910fc3b) - Unofficial documentation
- [Public ESPN API Repo](https://github.com/pseudo-r/Public-ESPN-API) - Community reverse-engineered endpoints
- Your backtesting framework: `src/simulations/backtesting.py`

## Support

If you encounter issues:
1. Run the test suite: `python scripts/test_data_pipeline.py`
2. Check the validation report for data quality issues
3. Review logs for API errors
4. Try with `--csv-only` mode first to isolate database issues
