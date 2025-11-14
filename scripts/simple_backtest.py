#!/usr/bin/env python3
"""
Simple Backtesting Script
=========================
Basic backtesting that works with the actual model interfaces.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

load_dotenv()

# Load games from database
db_url = os.getenv('DATABASE_URL', 'postgresql://hyder@localhost:5432/gamelens_ai')
engine = create_engine(db_url)

query = """
SELECT *
FROM games_history
WHERE league = 'NBA'
    AND game_date >= '2022-10-01'
    AND status = 'final'
    AND home_score IS NOT NULL
    AND away_score IS NOT NULL
ORDER BY game_date
"""

print("Loading games from database...")
games_df = pd.read_sql(query, engine)
print(f"✅ Loaded {len(games_df):,} games")

# Simple walk-forward validation
train_size = 500
test_size = 100

# Initialize Elo model
from src.simulations.enhanced_elo_model import EnhancedEloModel
elo_model = EnhancedEloModel(sport='NBA')

correct = 0
total = 0
predictions = []

print(f"\n🏀 Running Simple Backtest")
print(f"Training window: {train_size} games")
print(f"Test window: {test_size} games\n")

for start_idx in range(0, len(games_df) - train_size - test_size, test_size):
    # Training data
    train_df = games_df.iloc[start_idx:start_idx + train_size]
    test_df = games_df.iloc[start_idx + train_size:start_idx + train_size + test_size]

    # Re-initialize model for this window
    elo_model = EnhancedEloModel(sport='NBA')

    # Train: Update ratings on training games
    for _, game in train_df.iterrows():
        home_rating = elo_model.get_rating(str(game['home_team_id']))
        away_rating = elo_model.get_rating(str(game['away_team_id']))

        new_home, new_away = elo_model.update_ratings(
            rating_home=home_rating,
            rating_away=away_rating,
            score_home=int(game['home_score']),
            score_away=int(game['away_score']),
            neutral_site=bool(game.get('neutral_site', False))
        )

        # Store updated ratings
        elo_model.ratings[str(game['home_team_id'])] = new_home
        elo_model.ratings[str(game['away_team_id'])] = new_away

    # Test: Make predictions
    window_correct = 0
    for _, game in test_df.iterrows():
        home_rating = elo_model.get_rating(str(game['home_team_id']))
        away_rating = elo_model.get_rating(str(game['away_team_id']))

        # Predict home win probability
        home_win_prob = elo_model.win_probability(home_rating, away_rating)

        # Actual result
        actual_home_win = game['home_score'] > game['away_score']

        # Prediction (>50% means predict home win)
        predicted_home_win = home_win_prob > 0.5

        if predicted_home_win == actual_home_win:
            window_correct += 1
            correct += 1

        total += 1
        predictions.append({
            'date': game['game_date'],
            'home_team': game['home_team_name'],
            'away_team': game['away_team_name'],
            'predicted_prob': home_win_prob,
            'actual_win': actual_home_win,
            'correct': predicted_home_win == actual_home_win
        })

    window_acc = window_correct / len(test_df) if len(test_df) > 0 else 0
    print(f"Window {start_idx//test_size + 1}: {window_correct}/{len(test_df)} = {window_acc:.1%}")

    if total >= 500:  # Limit to 500 predictions for quick test
        break

overall_accuracy = correct / total if total > 0 else 0

print(f"\n{'='*60}")
print(f"BACKTEST RESULTS")
print(f"{'='*60}")
print(f"Model: Elo")
print(f"Total Predictions: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {overall_accuracy:.2%}")
print(f"{'='*60}\n")

# Calculate Brier score
preds_df = pd.DataFrame(predictions)
brier = np.mean((preds_df['predicted_prob'] - preds_df['actual_win'].astype(float))**2)
print(f"Brier Score: {brier:.4f}")

# Save results
output_file = f"backtest_results/simple_elo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
Path("backtest_results").mkdir(exist_ok=True)
preds_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")
