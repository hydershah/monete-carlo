"""
Hybrid prediction system combining Monte Carlo, Elo, and GPT analysis.
"""

from typing import Dict, Optional, Any
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session

from .monte_carlo import MonteCarloSimulator
from .elo_model import EloModel, get_elo_model
from .poisson_model import PoissonModel
from .gpt_analyzer import GPTAnalyzer
from ..models import Game, Team, Prediction, get_db_context


class HybridPredictor:
    """
    Hybrid prediction system that combines:
    - Monte Carlo simulations
    - Elo ratings
    - Poisson distribution (for low-scoring sports)
    - GPT qualitative analysis

    Weights and combines predictions for optimal accuracy.
    """

    def __init__(
        self,
        use_gpt: bool = True,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize hybrid predictor.

        Args:
            use_gpt: Whether to use GPT analysis (costs API tokens)
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        self.use_gpt = use_gpt

        # Initialize models
        self.monte_carlo = MonteCarloSimulator(n_simulations, random_seed)
        self.poisson = PoissonModel()
        self.gpt = GPTAnalyzer() if use_gpt else None

        # Elo models by sport (initialized on demand)
        self.elo_models = {}

    def get_elo_model(self, sport: str) -> EloModel:
        """Get or create Elo model for sport."""
        if sport not in self.elo_models:
            self.elo_models[sport] = get_elo_model(sport)
        return self.elo_models[sport]

    def predict_game(
        self,
        game: Game,
        db: Session,
        save_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Make comprehensive prediction for a game.

        Args:
            game: Game object from database
            db: Database session
            save_to_db: Whether to save prediction to database

        Returns:
            Comprehensive prediction dictionary
        """
        logger.info(
            f"Predicting {game.away_team.name} @ {game.home_team.name} "
            f"({game.sport.upper()})"
        )

        # Get Elo model for this sport
        elo_model = self.get_elo_model(game.sport)

        # Get Elo ratings
        home_elo = elo_model.get_or_create_rating(
            game.home_team_id,
            game.sport,
            db
        )
        away_elo = elo_model.get_or_create_rating(
            game.away_team_id,
            game.sport,
            db
        )

        # Make Elo prediction
        elo_prediction = elo_model.predict_game(home_elo, away_elo)

        # Run Monte Carlo simulation
        # For now, use Elo-based expected scores
        # In production, would use team stats
        mc_result = self._run_monte_carlo_for_game(
            game,
            elo_prediction,
            db
        )

        # Get GPT analysis if enabled
        gpt_analysis = None
        gpt_adjustment = 0.0
        if self.use_gpt and self.gpt:
            gpt_analysis = self._get_gpt_analysis(game, db)
            gpt_adjustment = gpt_analysis.get("adjustment", 0.0)

        # Combine predictions
        combined = self._combine_predictions(
            elo_prediction,
            mc_result,
            gpt_adjustment
        )

        # Build final prediction
        prediction_data = {
            "game_id": game.id,
            "model_version": "hybrid_v1.0",
            "prediction_date": datetime.now(),

            # Quantitative predictions
            "home_win_probability": elo_prediction["home_win_probability"],
            "away_win_probability": elo_prediction["away_win_probability"],

            # Monte Carlo results
            "predicted_home_score": mc_result.get("expected_home"),
            "predicted_away_score": mc_result.get("expected_away"),

            # Elo ratings
            "elo_home": home_elo,
            "elo_away": away_elo,
            "monte_carlo_iterations": self.monte_carlo.n_simulations,

            # GPT analysis
            "gpt_analysis": str(gpt_analysis) if gpt_analysis else None,
            "gpt_confidence": gpt_analysis.get("confidence") if gpt_analysis else None,
            "gpt_adjustment": gpt_adjustment,

            # Final combined prediction
            "final_home_win_prob": combined["home_win_prob"],
            "final_away_win_prob": combined["away_win_prob"],
            "confidence_level": combined["confidence"],

            # Metadata
            "metadata": {
                "elo_prediction": elo_prediction,
                "monte_carlo_summary": {
                    "home_win_pct": mc_result.get("home_win_pct"),
                    "away_win_pct": mc_result.get("away_win_pct"),
                    "simulations": self.monte_carlo.n_simulations,
                },
                "gpt_used": self.use_gpt,
            }
        }

        # Save to database if requested
        if save_to_db:
            prediction_obj = Prediction(**prediction_data)
            db.add(prediction_obj)
            db.commit()
            logger.info(f"Prediction saved to database (ID: {prediction_obj.id})")

        # Return full prediction
        return {
            "game": {
                "id": game.id,
                "home_team": game.home_team.name,
                "away_team": game.away_team.name,
                "date": game.game_date.isoformat() if game.game_date else None,
            },
            "prediction": prediction_data,
            "recommendation": self._generate_recommendation(combined, game),
        }

    def _run_monte_carlo_for_game(
        self,
        game: Game,
        elo_prediction: Dict[str, float],
        db: Session
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for a game."""
        # For NBA and high-scoring sports, use normal distribution
        if game.sport.lower() in ["nba", "nfl"]:
            # Estimate scores from Elo (rough approximation)
            # In production, would use actual team stats
            league_avg = {"nba": 112, "nfl": 23}. get(game.sport.lower(), 100)

            # Convert Elo difference to expected score difference
            elo_diff = elo_prediction["elo_difference"]
            spread = elo_prediction["estimated_spread"]

            # Estimate scores
            total_score = league_avg * 2
            home_score = (total_score + spread) / 2
            away_score = (total_score - spread) / 2

            result = self.monte_carlo.simulate_nba_game(
                home_ppg=home_score,
                away_ppg=away_score,
            )

            return {
                "expected_home": result.average_home_score,
                "expected_away": result.average_away_score,
                "home_win_pct": result.home_win_probability,
                "away_win_pct": result.away_win_probability,
            }

        else:
            # For soccer, hockey use Poisson
            # Default to moderate scoring
            home_expected = 2.5
            away_expected = 1.8

            result = self.monte_carlo.simulate_poisson_game(
                home_expected,
                away_expected
            )

            return {
                "expected_home": result.average_home_score,
                "expected_away": result.average_away_score,
                "home_win_pct": result.home_win_probability,
                "away_win_pct": result.away_win_probability,
                "draw_pct": result.draw_probability,
            }

    def _get_gpt_analysis(
        self,
        game: Game,
        db: Session
    ) -> Optional[Dict[str, Any]]:
        """Get GPT qualitative analysis for a game."""
        try:
            context = {
                "home_record": game.home_team_record or "",
                "away_record": game.away_team_record or "",
                # Add more context as available
            }

            analysis = self.gpt.analyze_matchup(
                home_team=game.home_team.name,
                away_team=game.away_team.name,
                sport=game.sport.upper(),
                game_date=game.game_date.strftime("%Y-%m-%d") if game.game_date else "",
                additional_context=context
            )

            return analysis

        except Exception as e:
            logger.warning(f"GPT analysis failed: {e}")
            return None

    def _combine_predictions(
        self,
        elo_pred: Dict[str, float],
        mc_result: Dict[str, Any],
        gpt_adjustment: float
    ) -> Dict[str, Any]:
        """
        Combine predictions from multiple models.

        Weighting:
        - 40% Elo
        - 40% Monte Carlo
        - 20% GPT adjustment

        Args:
            elo_pred: Elo prediction
            mc_result: Monte Carlo results
            gpt_adjustment: GPT probability adjustment

        Returns:
            Combined prediction
        """
        # Base probabilities (weighted average of Elo and MC)
        elo_weight = 0.5
        mc_weight = 0.5

        base_home_prob = (
            elo_weight * elo_pred["home_win_probability"] +
            mc_weight * mc_result.get("home_win_pct", elo_pred["home_win_probability"])
        )

        # Apply GPT adjustment (conservative, max ±10%)
        gpt_adjustment = max(min(gpt_adjustment, 0.10), -0.10)
        final_home_prob = base_home_prob + gpt_adjustment

        # Ensure probabilities are valid
        final_home_prob = max(min(final_home_prob, 0.99), 0.01)
        final_away_prob = 1.0 - final_home_prob

        # Calculate confidence
        # Higher confidence when models agree
        elo_mc_diff = abs(
            elo_pred["home_win_probability"] - mc_result.get("home_win_pct", 0.5)
        )
        model_agreement = 1.0 - elo_mc_diff
        confidence = model_agreement * 0.8  # Max 80% confidence

        return {
            "home_win_prob": final_home_prob,
            "away_win_prob": final_away_prob,
            "confidence": confidence,
            "gpt_adjustment_applied": gpt_adjustment,
        }

    def _generate_recommendation(
        self,
        prediction: Dict[str, Any],
        game: Game
    ) -> str:
        """Generate betting recommendation based on prediction."""
        home_prob = prediction["home_win_prob"]
        confidence = prediction["confidence"]

        # Get odds if available
        if game.odds_data and "theodds" in game.odds_data:
            # Find moneyline odds
            # In production, would parse odds properly
            pass

        # Simple recommendation logic
        if confidence < 0.6:
            return "PASS - Low confidence"
        elif home_prob > 0.65:
            return f"LEAN HOME - {game.home_team.name} ({home_prob:.1%})"
        elif home_prob < 0.35:
            return f"LEAN AWAY - {game.away_team.name} ({1-home_prob:.1%})"
        else:
            return "PASS - Too close to call"

    def predict_multiple_games(
        self,
        games: list[Game],
        db: Session,
        save_to_db: bool = True
    ) -> list[Dict[str, Any]]:
        """
        Predict multiple games efficiently.

        Args:
            games: List of Game objects
            db: Database session
            save_to_db: Whether to save predictions

        Returns:
            List of prediction dictionaries
        """
        predictions = []

        for game in games:
            try:
                prediction = self.predict_game(game, db, save_to_db)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting game {game.id}: {e}")
                continue

        logger.info(f"Predicted {len(predictions)} games")
        return predictions


def predict_todays_games(sport: str = "nba", use_gpt: bool = True):
    """
    Convenience function to predict all today's games for a sport.

    Args:
        sport: Sport identifier
        use_gpt: Whether to use GPT analysis

    Returns:
        List of predictions
    """
    predictor = HybridPredictor(use_gpt=use_gpt)

    with get_db_context() as db:
        # Get today's games
        from sqlalchemy import and_, cast, Date
        from datetime import date

        today = date.today()

        games = db.query(Game).filter(
            and_(
                Game.sport == sport.lower(),
                cast(Game.game_date, Date) == today,
                Game.status.in_(["scheduled", "pre"])
            )
        ).all()

        logger.info(f"Found {len(games)} games for {sport.upper()} today")

        if not games:
            logger.warning("No games found for today")
            return []

        # Make predictions
        predictions = predictor.predict_multiple_games(games, db, save_to_db=True)

        return predictions


if __name__ == "__main__":
    # Test hybrid predictor
    from dotenv import load_dotenv
    load_dotenv()

    print("\n=== Hybrid Predictor Example ===\n")

    # This would typically be run after ingesting game data
    print("To test, first run data ingestion:")
    print("  python -m src.data.ingestion")
    print("\nThen predict games:")
    print("  python -m src.simulations.hybrid_predictor")

    # predictions = predict_todays_games("nba", use_gpt=False)
    # for pred in predictions:
    #     print(f"\n{pred['game']['away_team']} @ {pred['game']['home_team']}")
    #     print(f"Prediction: {pred['recommendation']}")
    #     print(f"Confidence: {pred['prediction']['confidence_level']:.1%}")
