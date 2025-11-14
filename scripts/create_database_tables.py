#!/usr/bin/env python3
"""
Create database tables for GameLens.ai backtesting system
Bypasses the conflicting models in src/models/__init__.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct import to avoid __init__.py conflicts
from sqlalchemy import create_engine
import importlib.util

# Load database_schema.py directly
schema_path = Path(__file__).parent.parent / "src" / "models" / "database_schema.py"
spec = importlib.util.spec_from_file_location("database_schema", schema_path)
database_schema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(database_schema)

DATABASE_URL = 'postgresql://hyder@localhost:5432/gamelens_ai'

print("Creating GameLens.ai database schema...")
print(f"Database: {DATABASE_URL}")

engine = create_engine(DATABASE_URL)

# Create all tables
database_schema.create_all_tables(engine)

print("✅ All tables created successfully!")
print()
print("Tables created:")
print("  • games_history - Historical games since 2010")
print("  • team_stats_daily - Daily team statistics snapshots")
print("  • odds_history - Historical betting lines")
print("  • team_ratings_history - Elo/Pythagorean ratings over time")
print("  • predictions_log - All system predictions")
print("  • model_performance - Aggregated model metrics")
print("  • clv_analytics - CLV tracking")
print("  • calibration_metrics - Calibration performance")
