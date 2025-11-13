"""
Example script demonstrating programmatic usage of the prediction system.
"""

from dotenv import load_dotenv
load_dotenv()

from src.models import get_db_context, Game, Team, Prediction
from src.data.espn_client import ESPNClient
from src.data.ingestion import DataIngestion
from src.simulations.hybrid_predictor import HybridPredictor
from src.simulations.monte_carlo import MonteCarloSimulator
from src.simulations.elo_model import get_elo_model
from datetime import datetime

print("\n" + "="*60)
print("  Sports Prediction System - Example Usage")
print("="*60 + "\n")

# Example 1: Fetch and display today's NBA games
print("Example 1: Fetching NBA games")
print("-" * 60)

espn = ESPNClient()
games = espn.get_todays_games("nba")

print(f"Found {len(games)} NBA games today:\n")
for i, game in enumerate(games[:3], 1):  # Show first 3
    print(f"{i}. {game['away_team_abbr']} @ {game['home_team_abbr']}")
    print(f"   Status: {game['status']}")
    if game['home_score'] > 0:
        print(f"   Score: {game['away_score']} - {game['home_score']}")
    print()

# Example 2: Run Monte Carlo simulation
print("\nExample 2: Monte Carlo Simulation")
print("-" * 60)

simulator = MonteCarloSimulator(n_simulations=10000, random_seed=42)

# Simulate Lakers (112 PPG) vs Celtics (110 PPG)
result = simulator.simulate_nba_game(
    home_ppg=112,
    away_ppg=110,
    home_defensive_rating=108,
    away_defensive_rating=106
)

print("Lakers vs Celtics Simulation (10,000 iterations):\n")
print(f"  Lakers win:   {result.home_win_probability:.1%}")
print(f"  Celtics win:  {result.away_win_probability:.1%}")
print(f"\n  Expected Score:")
print(f"    Lakers:  {result.average_home_score:.1f}")
print(f"    Celtics: {result.average_away_score:.1f}")

# Example 3: Elo ratings
print("\n\nExample 3: Elo Rating Predictions")
print("-" * 60)

elo_model = get_elo_model("nba")

# Predict game with Elo ratings
prediction = elo_model.predict_game(
    home_rating=1600,  # Lakers
    away_rating=1550   # Celtics
)

print("Prediction using Elo ratings:\n")
print(f"  Home team (1600 Elo): {prediction['home_win_probability']:.1%}")
print(f"  Away team (1550 Elo): {prediction['away_win_probability']:.1%}")
print(f"  Estimated spread:     {prediction['estimated_spread']:.1f} points")

# Example 4: Database interaction
print("\n\nExample 4: Database Queries")
print("-" * 60)

try:
    with get_db_context() as db:
        # Count teams, games, predictions
        team_count = db.query(Team).count()
        game_count = db.query(Game).count()
        pred_count = db.query(Prediction).count()

        print(f"\nDatabase Statistics:")
        print(f"  Teams:       {team_count}")
        print(f"  Games:       {game_count}")
        print(f"  Predictions: {pred_count}")

        # Show recent predictions
        if pred_count > 0:
            recent_pred = db.query(Prediction).order_by(
                Prediction.prediction_date.desc()
            ).first()

            if recent_pred and recent_pred.game:
                print(f"\nMost Recent Prediction:")
                game = recent_pred.game
                print(f"  {game.away_team.name} @ {game.home_team.name}")
                print(f"  Home win prob: {recent_pred.final_home_win_prob:.1%}")
                print(f"  Confidence: {recent_pred.confidence_level:.1%}")

except Exception as e:
    print(f"  Database not initialized. Run: python -m src.cli init")
    print(f"  Error: {e}")

# Example 5: Full prediction workflow
print("\n\nExample 5: Full Prediction Workflow")
print("-" * 60)

print("""
# To run a complete prediction workflow:

1. Initialize database:
   python -m src.cli init

2. Fetch game data:
   python -m src.cli fetch --sport nba --date today --with-odds

3. Make predictions:
   python -m src.cli predict --sport nba

4. View results:
   python -m src.cli results --sport nba --days 1

# Or use programmatically:

from src.simulations.hybrid_predictor import predict_todays_games

predictions = predict_todays_games("nba", use_gpt=True)

for pred in predictions:
    print(f"{pred['game']['away_team']} @ {pred['game']['home_team']}")
    print(f"Prediction: {pred['recommendation']}")
""")

print("\n" + "="*60)
print("  For more examples, see GETTING_STARTED.md")
print("="*60 + "\n")
