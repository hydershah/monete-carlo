# Backtesting Data Setup - Complete! ✅

## What Was Created

Your backtesting system is now ready with a complete historical data pipeline powered by ESPN API.

### 1. **Historical Data Fetcher** ([src/data/historical_data_fetcher.py](src/data/historical_data_fetcher.py))

A robust data fetcher that:
- ✅ Fetches historical game data from ESPN API (free, no API key required)
- ✅ Supports NBA, NFL, MLB, NHL
- ✅ Handles multiple seasons automatically
- ✅ Validates data quality
- ✅ Saves to both database and CSV
- ✅ Formats data for your backtesting framework

### 2. **Data Ingestion Script** ([scripts/ingest_historical_data.py](scripts/ingest_historical_data.py))

Command-line tool with features:
- ✅ Batch data ingestion
- ✅ Multiple league support
- ✅ Progress tracking with visual feedback
- ✅ CSV backup capability
- ✅ Comprehensive validation reporting
- ✅ Error recovery

### 3. **Test Suite** ([scripts/test_data_pipeline.py](scripts/test_data_pipeline.py))

Complete testing that validates:
- ✅ ESPN API connectivity
- ✅ Data fetching and parsing
- ✅ Data format compatibility
- ✅ Quality checks

### 4. **Documentation** ([DATA_INGESTION_GUIDE.md](DATA_INGESTION_GUIDE.md))

Comprehensive guide covering:
- ✅ Quick start instructions
- ✅ API details and rate limits
- ✅ Data format specifications
- ✅ Backtesting integration
- ✅ Troubleshooting tips

---

## Quick Start

### Step 1: Test the Pipeline (Already Done!)
```bash
python scripts/test_data_pipeline.py
```
**Status:** ✅ All tests passed! 194 NBA games fetched successfully.

### Step 2: Ingest Historical Data

#### Option A: Quick Start (3 NBA Seasons - Recommended)
```bash
python scripts/ingest_historical_data.py
```
- **Time:** ~10-15 minutes
- **Games:** ~3,690 NBA games (3 seasons)
- **Output:** Database + CSV backup

#### Option B: Multiple Leagues
```bash
python scripts/ingest_historical_data.py --league nba nfl --seasons 3
```

#### Option C: CSV Only (No Database)
```bash
python scripts/ingest_historical_data.py --csv-only --output-dir ./data/historical
```

### Step 3: Use the Data for Backtesting

```python
from src.data.historical_data_fetcher import HistoricalDataFetcher
from src.simulations.backtesting import WalkForwardAnalyzer
from sklearn.linear_model import LogisticRegression

# Load historical data
fetcher = HistoricalDataFetcher()
games = fetcher.load_from_csv('data/historical/nba_historical_3seasons.csv')
df = fetcher.get_games_dataframe(games)

# Run backtesting
analyzer = WalkForwardAnalyzer(
    train_window=500,  # 500 games for training
    test_window=100,   # 100 games for testing
    step_size=50       # Roll forward 50 games each iteration
)

model = LogisticRegression()
results = analyzer.analyze(model, df)

# View results
from src.simulations.backtesting import create_backtest_report
print(create_backtest_report(results))
```

---

## Data Sources

### ESPN API (Primary)
- **URL:** `https://site.api.espn.com/apis/site/v2/sports`
- **Cost:** FREE (no API key required)
- **Coverage:** NBA, NFL, MLB, NHL, Soccer
- **Historical:** Multiple seasons available
- **Rate Limit:** ~1-2 seconds recommended
- **Status:** ✅ Tested and working

### Data Available
- Game scores and results
- Team information
- Game dates and venues
- Betting lines (when available)
- Season and playoff indicators

---

## Sample Data Fetched

The test successfully fetched:
- **194 NBA games** from the last 30 days
- **Date Range:** October 15, 2025 - November 14, 2025
- **Completed Games:** All 194
- **Data Quality:** ✅ No duplicates, valid scores, proper formatting
- **CSV Saved:** [data/test/sample_games.csv](data/test/sample_games.csv)

**Sample Game Data:**
```
game_date                  home_team_name        away_team_name         home_score  away_score
2025-10-15 23:00:00+00:00  Charlotte Hornets     Houston Rockets        145         116
2025-10-15 23:30:00+00:00  Boston Celtics        Philadelphia 76ers     110         108
2025-10-16 02:00:00+00:00  Sacramento Kings      Minnesota Timberwolves 91          109
```

---

## What's Next?

### 1. Ingest Full Historical Data
```bash
# Recommended: Fetch 3 seasons of NBA data
python scripts/ingest_historical_data.py --seasons 3

# This will take ~10-15 minutes and fetch ~3,690 games
```

### 2. Run Your First Backtest

Once data is ingested, you can:
- Test your Monte Carlo simulations with real historical data
- Validate your Elo model against actual results
- Test your ensemble predictions
- Calculate CLV (Closing Line Value) metrics
- Calibrate your models

### 3. Automate Daily Updates

Set up a cron job to fetch daily results:
```bash
# Add to crontab: Run daily at 6 AM
0 6 * * * cd /Users/hyder/monete\ carlo && python scripts/ingest_historical_data.py --days 1
```

---

## Files Created

```
monete carlo/
├── src/data/
│   ├── historical_data_fetcher.py     # Main data fetcher class
│   ├── espn_client.py                 # ESPN API client (enhanced)
│   └── goalserve_client.py            # GoalServe client (existing)
│
├── scripts/
│   ├── ingest_historical_data.py      # CLI tool for data ingestion
│   └── test_data_pipeline.py          # Test suite
│
├── data/
│   ├── historical/                    # Historical data CSVs (created by script)
│   └── test/
│       └── sample_games.csv           # Test data (194 games)
│
├── DATA_INGESTION_GUIDE.md            # Comprehensive documentation
└── BACKTESTING_DATA_SETUP.md          # This file
```

---

## Performance & Estimates

### Ingestion Time (1s rate limit)

| League | Games/Season | Seasons | Estimated Time |
|--------|--------------|---------|----------------|
| NBA    | 1,230        | 3       | 10-15 min      |
| NFL    | 285          | 3       | 3-5 min        |
| MLB    | 2,430        | 3       | 20-25 min      |
| NHL    | 1,312        | 3       | 12-15 min      |

### Storage Estimates

| League | Seasons | Games   | CSV Size | Database Size |
|--------|---------|---------|----------|---------------|
| NBA    | 3       | ~3,690  | ~2.5 MB  | ~15 MB        |
| NFL    | 3       | ~855    | ~600 KB  | ~4 MB         |
| MLB    | 3       | ~7,290  | ~5 MB    | ~30 MB        |
| NHL    | 3       | ~3,936  | ~2.7 MB  | ~16 MB        |

---

## Key Features

### 1. Data Quality Validation
```
✅ No duplicate games
✅ Valid scores (no negative values)
✅ Proper date formatting
✅ Timezone handling
✅ Missing data reporting
```

### 2. Backtesting Integration
The fetched data is formatted to work directly with your backtesting framework:
- `game_id`: Unique identifier
- `game_date`: Datetime for temporal ordering
- `home_team_id`, `away_team_id`: Team identifiers
- `home_score`, `away_score`: Final scores
- `league`, `season`: Grouping fields
- `spread`, `total`: Betting lines (when available)

### 3. Error Handling
- Automatic retries on API failures
- Rate limiting to avoid blocks
- CSV backup if database fails
- Graceful degradation

### 4. Flexibility
- CSV-only mode for bulk operations
- Custom date ranges
- Multiple leagues simultaneously
- Incremental updates

---

## Troubleshooting

### If ESPN API is slow:
```bash
# Increase rate limit (risky)
python scripts/ingest_historical_data.py --rate-limit 2.0
```

### If database connection fails:
```bash
# Use CSV-only mode
python scripts/ingest_historical_data.py --csv-only
```

### If you need to resume:
```python
# Load from CSV and re-insert to database
fetcher = HistoricalDataFetcher()
games = fetcher.load_from_csv('data/historical/nba_historical_3seasons.csv')
fetcher.save_to_database(games)
```

---

## Summary

🎉 **Your backtesting system is ready!**

✅ Data pipeline tested and validated
✅ 194 NBA games successfully fetched
✅ Data format compatible with backtesting framework
✅ ESPN API confirmed working
✅ Comprehensive documentation provided

**Next Action:** Run `python scripts/ingest_historical_data.py` to fetch 3 seasons of NBA data (~10-15 minutes)

---

## Resources

- **ESPN API Docs:** [Unofficial Gist](https://gist.github.com/akeaswaran/b48b02f1c94f873c6655e7129910fc3b)
- **Your Backtesting Framework:** [src/simulations/backtesting.py](src/simulations/backtesting.py)
- **Database Schema:** [src/models/database_schema.py](src/models/database_schema.py)
- **Data Manager:** [src/data/data_manager.py](src/data/data_manager.py)

---

**Created:** November 14, 2025
**Test Status:** ✅ All tests passed
**Sample Data:** 194 NBA games (last 30 days)
**Ready for:** Production backtesting
