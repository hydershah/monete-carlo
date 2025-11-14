# GameLens.ai Data Pipeline Documentation

## Overview

Complete data infrastructure for fetching, storing, and serving sports prediction data with sub-200ms response times.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: DATA SOURCES                                       │
├─────────────────────────────────────────────────────────────┤
│ • Goalserve API (historical data since 2010)               │
│ • ESPN API (current season, live scores)                   │
│ • TheOdds API (current odds, optional)                     │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: STORAGE                                            │
├─────────────────────────────────────────────────────────────┤
│ PostgreSQL (Cold Storage)                                   │
│ • games_history - All historical games (2010-present)      │
│ • team_stats_daily - Daily team statistics snapshots       │
│ • odds_history - Historical betting lines                  │
│ • team_ratings_history - Elo/Pythagorean over time        │
│ • predictions_log - Model predictions & outcomes           │
│ • model_performance - Aggregated model metrics             │
│                                                              │
│ Redis (Hot Cache)                                           │
│ • team_stats:{league}:{team_id} - TTL 6h                  │
│ • prediction:{params} - TTL 15 min                         │
│ • odds:{game_id} - TTL 5 min                              │
│ • model_weights:{league} - TTL 24h                        │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: DATA MANAGER (Unified Interface)                  │
├─────────────────────────────────────────────────────────────┤
│ • get_team_stats() - Redis → DB → API                     │
│ • get_prediction_data() - Complete data for predictions    │
│ • get_team_games() - Recent game history                   │
│ • warmup_cache() - Pre-load before game day                │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 4: PREDICTION MODELS                                  │
├─────────────────────────────────────────────────────────────┤
│ Uses data from DataManager to make predictions             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup Database

```bash
# Set environment variables
export DATABASE_URL="postgresql://user:password@localhost:5432/gamelens"
export REDIS_HOST="localhost"
export REDIS_PORT=6379
export GOALSERVE_API_KEY="your_api_key"

# Create tables
python3 -c "from src.models.database_schema import create_all_tables; from sqlalchemy import create_engine; create_all_tables(create_engine('$DATABASE_URL'))"
```

### 2. Backfill Historical Data (One-Time)

```bash
# Backfill NBA data from 2020-2024
python3 scripts/backfill_historical_data.py \
    --league NBA \
    --start-year 2020 \
    --end-year 2024 \
    --create-tables

# Backfill all leagues (takes several hours)
python3 scripts/backfill_historical_data.py \
    --league all \
    --start-year 2020 \
    --end-year 2024 \
    --resume  # Resume if interrupted
```

**Progress Tracking:**
- Script saves progress to `backfill_progress.json`
- Use `--resume` flag to skip already processed months
- Estimated time: ~2-4 hours for 5 years of data across all leagues

### 3. Setup Daily Jobs

Add to your crontab:

```bash
# Update yesterday's completed games (3 AM daily)
0 3 * * * cd /path/to/project && python3 scripts/daily_update_job.py --job update_games

# Refresh team stats cache (every 6 hours)
0 */6 * * * cd /path/to/project && python3 scripts/daily_update_job.py --job refresh_cache

# Update live odds (every 15 minutes, 8 AM - 11 PM)
*/15 8-23 * * * cd /path/to/project && python3 scripts/daily_update_job.py --job update_odds
```

Or run all jobs manually:

```bash
python3 scripts/daily_update_job.py --job all
```

### 4. Use in Your Code

```python
from src.data.data_manager import get_data_manager

# Initialize data manager (singleton)
dm = get_data_manager()

# Get team stats (checks Redis → DB → API)
lakers_stats = dm.get_team_stats('lakers', 'NBA')
print(f"Lakers Elo: {lakers_stats['elo_rating']}")

# Get complete prediction data
prediction_data = dm.get_prediction_data(
    home_team_id='celtics',
    away_team_id='lakers',
    league='NBA',
    spread=-5.5,
    total=218.5
)

# Now use with your models
from src.simulations.meta_ensemble import MetaEnsemble

ensemble = MetaEnsemble(sport='NBA')
prediction = ensemble.predict_game(
    home_rating=prediction_data.home_elo,
    away_rating=prediction_data.away_elo,
    home_stats=prediction_data.home_stats,
    away_stats=prediction_data.away_stats,
    spread=prediction_data.spread,
    total=prediction_data.total
)

print(f"Home win probability: {prediction['home_win_probability']:.1%}")
```

## Performance Characteristics

### Response Times

| Operation | First Request | Cached Request | Method |
|-----------|--------------|----------------|--------|
| **Get team stats** | 10-20ms | 1-2ms | Redis cache |
| **Get team games** | 50-100ms | N/A | PostgreSQL query |
| **Complete prediction** | 150-200ms | 2-5ms | DataManager + cache |
| **10,000 simulations** | 100ms | N/A | Monte Carlo engine |

### Storage

| Data Type | Size | Retention |
|-----------|------|-----------|
| **Historical games** | ~10GB | Permanent (2010-present) |
| **Team stats snapshots** | ~1GB | Daily snapshots |
| **Odds history** | ~5GB | Permanent |
| **Predictions log** | ~500MB/year | Permanent |
| **Redis cache** | ~100MB | 5min - 24h TTL |

## Data Flow Examples

### Example 1: Morning Cache Warmup

```python
from src.data.data_manager import get_data_manager

dm = get_data_manager()

# Get today's games
todays_games = dm.get_todays_games('NBA')

# Pre-warm cache for all teams playing today
dm.warmup_cache_for_games(todays_games)

# Now all prediction requests will hit cache (1-2ms response)
```

### Example 2: Making a Prediction (Production)

```python
from src.data.data_manager import get_data_manager
from src.simulations.meta_ensemble import MetaEnsemble

# Step 1: Get data (checks cache first)
dm = get_data_manager()
data = dm.get_prediction_data('lakers', 'celtics', 'NBA', -5.5, 218.5)

# Step 2: Run ensemble prediction
ensemble = MetaEnsemble(sport='NBA')
prediction = ensemble.predict_game(
    home_rating=data.home_elo,
    away_rating=data.away_elo,
    home_stats=data.home_stats,
    away_stats=data.away_stats,
    spread=data.spread,
    total=data.total
)

# Step 3: Cache the prediction (15 min TTL)
# (Handled automatically by prediction API)

# Total time: ~150ms first request, ~2ms subsequent requests
```

### Example 3: Checking Cache Statistics

```python
from config.redis_config import get_cache

cache = get_cache()

stats = cache.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1f}%")
print(f"Total keys: {stats['total_keys']}")
print(f"Memory used: {stats['memory_used']}")
```

## Database Schema

### games_history

Primary table for all historical games.

```sql
SELECT * FROM games_history WHERE league = 'NBA' AND game_date = '2024-01-15';
```

**Key columns:**
- `game_id` (PK) - Unique game identifier
- `league` - NBA, NFL, MLB, NHL, etc.
- `home_team_id`, `away_team_id` - Team identifiers
- `home_score`, `away_score` - Final scores
- `spread`, `total` - Betting lines at game time
- `home_covered`, `went_over` - Betting outcomes

**Indexes:**
- `(league, game_date)` - Fast daily queries
- `(home_team_id, game_date)` - Team history queries
- `(season, league)` - Season-wide analysis

### team_stats_daily

Daily snapshots of team statistics.

```sql
SELECT * FROM team_stats_daily
WHERE team_id = 'lakers'
AND snapshot_date = CURRENT_DATE;
```

**Key columns:**
- `team_id`, `league`, `snapshot_date` - Composite key
- `wins`, `losses`, `win_pct` - Record
- `ppg`, `papg` - Scoring averages
- `elo_rating` - Current Elo rating
- `pythagorean_win_pct` - Expected win %

### odds_history

Historical betting lines and movements.

```sql
SELECT * FROM odds_history
WHERE game_id = 'game_123'
AND is_closing_line = true;
```

### predictions_log

All predictions made by the system.

```sql
SELECT model_name, AVG(CASE WHEN prediction_correct THEN 1 ELSE 0 END) as accuracy
FROM predictions_log
WHERE prediction_datetime >= NOW() - INTERVAL '7 days'
GROUP BY model_name;
```

## Redis Cache Keys

### Naming Convention

```
{prefix}:{league}:{identifier}:{params}
```

### Key Examples

```
team_stats:NBA:lakers                    # Lakers current stats
prediction:celtics:lakers:-5.5:218.5     # Specific game prediction
odds:game_12345                          # Current odds for game
model_weights:NBA                        # Ensemble weights for NBA
elo_rating:NBA:celtics                   # Celtics Elo rating
```

### TTL (Time-To-Live)

- **Team stats**: 6 hours (updated by background job)
- **Predictions**: 15 minutes (odds change frequently)
- **Live odds**: 5 minutes (during active betting)
- **Game info**: 24 hours (schedule doesn't change much)
- **Model weights**: 24 hours (updated daily)

## Monitoring & Maintenance

### Check Data Freshness

```python
from src.data.data_manager import get_data_manager
from datetime import date

dm = get_data_manager()

# Check if yesterday's games are in DB
yesterday = date.today() - timedelta(days=1)
games = dm.get_todays_games('NBA')  # Modify to accept date parameter

if not games:
    print("⚠️ Warning: No games found for yesterday. Run update job!")
```

### Clear Cache

```python
from config.redis_config import get_cache

cache = get_cache()

# Clear all predictions (force fresh calculations)
cache.clear_cache("prediction:*")

# Clear all team stats (force DB queries)
cache.clear_cache("team_stats:*")

# Clear everything
cache.clear_cache()
```

### Database Maintenance

```sql
-- Check database size
SELECT pg_size_pretty(pg_database_size('gamelens'));

-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Vacuum to reclaim space
VACUUM ANALYZE games_history;
```

## Troubleshooting

### Issue: Predictions are slow (>200ms)

**Check:**
1. Is Redis running? `redis-cli ping`
2. Is cache warming before game day? `dm.warmup_cache_for_games()`
3. Check cache hit rate: `cache.get_cache_stats()`

**Fix:**
```bash
# Restart Redis
redis-cli
> FLUSHALL
> exit

# Warm cache
python3 -c "from src.data.data_manager import get_data_manager; dm = get_data_manager(); games = dm.get_todays_games('NBA'); dm.warmup_cache_for_games(games)"
```

### Issue: Missing team stats

**Check:**
```python
dm = get_data_manager()
stats = dm.get_team_stats('lakers', 'NBA', force_refresh=True)
```

**Fix:**
```bash
# Run cache refresh job
python3 scripts/daily_update_job.py --job refresh_cache
```

### Issue: Historical data incomplete

**Check:**
```sql
SELECT league, COUNT(*) as games, MIN(game_date), MAX(game_date)
FROM games_history
GROUP BY league;
```

**Fix:**
```bash
# Resume backfill for missing data
python3 scripts/backfill_historical_data.py --league NBA --start-year 2020 --resume
```

## API Rate Limits

### Goalserve
- ~1 request per second recommended
- Backfill script includes automatic rate limiting
- Use `time.sleep(1)` between batch requests

### ESPN
- Unofficial API, no documented limits
- Use `rate_limit=0.5` (default in ESPNClient)
- Avoid parallel requests to same endpoint

### TheOdds API
- Free tier: 500 requests/month
- Paid tier: 10,000+ requests/month
- Check usage: `theodds_client.get_usage()`

## Production Checklist

- [ ] PostgreSQL database created and tables initialized
- [ ] Redis server running and accessible
- [ ] API keys set in environment variables
- [ ] Historical data backfilled (at least 3 years)
- [ ] Daily update jobs scheduled in cron
- [ ] Cache warmup runs before game day (8 AM)
- [ ] Monitoring alerts configured
- [ ] Database backups automated
- [ ] Log rotation configured
- [ ] Performance metrics tracked

## Next Steps

1. **Backfill historical data** - Run backfill script for your target leagues
2. **Setup cron jobs** - Automate daily updates
3. **Test data flow** - Verify cache → DB → API fallbacks work
4. **Monitor performance** - Track response times and cache hit rates
5. **Scale as needed** - Add read replicas, Redis cluster, etc.

## Support

For issues or questions:
- Check logs in `/var/log/gamelens/`
- Review cache stats with `get_cache().get_cache_stats()`
- Query database directly for data validation
- Test components individually before full integration
