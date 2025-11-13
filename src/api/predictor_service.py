"""
Standalone prediction service that doesn't require database.
Can analyze games directly from input data.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import time
from loguru import logger

from ..simulations.monte_carlo import MonteCarloSimulator
from ..simulations.elo_model import get_elo_model
from ..simulations.gpt_analyzer import GPTAnalyzer
from .schemas import GameDataInput, PredictionResponse


class PredictionService:
    """
    Standalone prediction service for analyzing games.

    Works without database - accepts game data and returns predictions.
    """

    def __init__(self):
        """Initialize prediction service."""
        self.gpt_analyzer = None
        self.monte_carlo = None

    def _init_gpt(self):
        """Lazy initialization of GPT analyzer."""
        if self.gpt_analyzer is None:
            try:
                self.gpt_analyzer = GPTAnalyzer()
            except Exception as e:
                logger.warning(f"GPT analyzer not available: {e}")

    def _init_monte_carlo(self, iterations: int = 10000):
        """Lazy initialization of Monte Carlo simulator."""
        if self.monte_carlo is None or self.monte_carlo.n_simulations != iterations:
            self.monte_carlo = MonteCarloSimulator(n_simulations=iterations)

    def predict_game(self, game_data: GameDataInput) -> PredictionResponse:
        """
        Predict game outcome from input data.

        Args:
            game_data: Game information and statistics

        Returns:
            PredictionResponse with predictions
        """
        start_time = time.time()

        logger.info(
            f"Predicting {game_data.away_team} @ {game_data.home_team} "
            f"({game_data.sport.upper()})"
        )

        models_used = []

        # 1. Run Monte Carlo simulation if enabled
        mc_result = None
        if game_data.use_monte_carlo:
            mc_result = self._run_monte_carlo(game_data)
            models_used.append("monte_carlo")

        # 2. Get GPT analysis if enabled
        gpt_analysis = None
        gpt_adjustment = 0.0
        if game_data.use_gpt:
            self._init_gpt()
            if self.gpt_analyzer:
                gpt_analysis = self._get_gpt_analysis(game_data)
                gpt_adjustment = gpt_analysis.get("adjustment", 0.0)
                models_used.append("gpt")

        # 3. Combine predictions
        combined = self._combine_predictions(
            mc_result=mc_result,
            gpt_adjustment=gpt_adjustment,
            game_data=game_data
        )

        # 4. Generate recommendation
        recommendation = self._generate_recommendation(
            combined,
            game_data
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms

        # Build response
        response = PredictionResponse(
            home_team=game_data.home_team,
            away_team=game_data.away_team,
            sport=game_data.sport,
            game_date=game_data.game_date,

            home_win_probability=combined["home_win_prob"],
            away_win_probability=combined["away_win_prob"],
            draw_probability=combined.get("draw_prob"),

            predicted_home_score=combined.get("expected_home_score"),
            predicted_away_score=combined.get("expected_away_score"),

            confidence=combined["confidence"],
            recommendation=recommendation,

            models_used=models_used,
            gpt_analysis=gpt_analysis.get("reasoning") if gpt_analysis else None,
            gpt_adjustment=gpt_adjustment,

            most_likely_score=combined.get("most_likely_score"),
            spread_analysis=combined.get("spread_analysis"),
            total_analysis=combined.get("total_analysis"),

            prediction_timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )

        logger.info(
            f"Prediction complete in {processing_time:.0f}ms - "
            f"{recommendation}"
        )

        return response

    def _run_monte_carlo(self, game_data: GameDataInput) -> Dict[str, Any]:
        """Run Monte Carlo simulation for the game."""
        self._init_monte_carlo(game_data.monte_carlo_iterations)

        # Determine if we have stats
        if game_data.home_ppg and game_data.away_ppg:
            # Use provided stats
            if game_data.sport.lower() in ["nba", "nfl"]:
                # High-scoring sports - use normal distribution
                result = self.monte_carlo.simulate_nba_game(
                    home_ppg=game_data.home_ppg,
                    away_ppg=game_data.away_ppg,
                    home_defensive_rating=game_data.home_defensive_rating or 110.0,
                    away_defensive_rating=game_data.away_defensive_rating or 110.0,
                    league_avg_ppg=self._get_league_avg(game_data.sport)
                )
            else:
                # Low-scoring sports - use Poisson
                # Convert to goals per game (rough estimate)
                home_gpg = game_data.home_ppg / 40 if game_data.home_ppg else 2.5
                away_gpg = game_data.away_ppg / 40 if game_data.away_ppg else 2.0

                result = self.monte_carlo.simulate_poisson_game(
                    home_gpg,
                    away_gpg
                )
        else:
            # Use default values based on sport
            league_avg = self._get_league_avg(game_data.sport)

            if game_data.sport.lower() in ["nba", "nfl"]:
                result = self.monte_carlo.simulate_nba_game(
                    home_ppg=league_avg * 1.05,  # Slight home advantage
                    away_ppg=league_avg * 0.95,
                )
            else:
                result = self.monte_carlo.simulate_poisson_game(2.5, 2.0)

        return {
            "home_win_prob": result.home_win_probability,
            "away_win_prob": result.away_win_probability,
            "draw_prob": result.draw_probability if hasattr(result, 'draw_probability') else 0.0,
            "expected_home": result.average_home_score,
            "expected_away": result.average_away_score,
        }

    def _get_gpt_analysis(self, game_data: GameDataInput) -> Optional[Dict[str, Any]]:
        """Get GPT analysis for the game."""
        try:
            context = {
                "home_record": game_data.home_record or "",
                "away_record": game_data.away_record or "",
                "injuries": game_data.injuries or "",
                "recent_form": game_data.recent_form or "",
                "head_to_head": game_data.head_to_head or "",
            }

            # Remove empty values
            context = {k: v for k, v in context.items() if v}

            analysis = self.gpt_analyzer.analyze_matchup(
                home_team=game_data.home_team,
                away_team=game_data.away_team,
                sport=game_data.sport.upper(),
                game_date=game_data.game_date or datetime.now().strftime("%Y-%m-%d"),
                additional_context=context if context else None
            )

            return analysis

        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            return None

    def _combine_predictions(
        self,
        mc_result: Optional[Dict[str, Any]],
        gpt_adjustment: float,
        game_data: GameDataInput
    ) -> Dict[str, Any]:
        """Combine predictions from multiple models."""

        if mc_result:
            # Use Monte Carlo as base
            base_home_prob = mc_result["home_win_prob"]
            expected_home = mc_result["expected_home"]
            expected_away = mc_result["expected_away"]
            draw_prob = mc_result.get("draw_prob", 0.0)
        else:
            # Fallback to 50-50 with slight home advantage
            base_home_prob = 0.52
            expected_home = self._get_league_avg(game_data.sport)
            expected_away = self._get_league_avg(game_data.sport) * 0.98
            draw_prob = 0.0

        # Apply GPT adjustment (conservative, max ±10%)
        gpt_adjustment = max(min(gpt_adjustment, 0.10), -0.10)
        final_home_prob = base_home_prob + gpt_adjustment

        # Ensure valid probabilities
        final_home_prob = max(min(final_home_prob, 0.99), 0.01)
        final_away_prob = 1.0 - final_home_prob - draw_prob

        # Calculate confidence (higher when adjustment is small)
        confidence = 1.0 - abs(gpt_adjustment) * 3  # Penalize large adjustments
        confidence = max(min(confidence, 0.95), 0.50)

        # Format most likely score
        most_likely = f"{int(round(expected_home))}-{int(round(expected_away))}"

        return {
            "home_win_prob": final_home_prob,
            "away_win_prob": final_away_prob,
            "draw_prob": draw_prob if draw_prob > 0.01 else None,
            "expected_home_score": expected_home,
            "expected_away_score": expected_away,
            "confidence": confidence,
            "most_likely_score": most_likely,
        }

    def _generate_recommendation(
        self,
        prediction: Dict[str, Any],
        game_data: GameDataInput
    ) -> str:
        """Generate betting recommendation."""
        home_prob = prediction["home_win_prob"]
        confidence = prediction["confidence"]

        # Check if we have odds to calculate value
        has_value = False
        if game_data.betting_odds:
            # TODO: Calculate if prediction vs odds shows value
            pass

        # Simple recommendation logic
        if confidence < 0.60:
            return "PASS - Low confidence"
        elif home_prob > 0.65:
            strength = "STRONG" if home_prob > 0.75 else "LEAN"
            return f"{strength} HOME - {game_data.home_team} ({home_prob:.1%})"
        elif home_prob < 0.35:
            strength = "STRONG" if home_prob < 0.25 else "LEAN"
            return f"{strength} AWAY - {game_data.away_team} ({1-home_prob:.1%})"
        else:
            return "PASS - Too close to call"

    def _get_league_avg(self, sport: str) -> float:
        """Get league average score for a sport."""
        averages = {
            "nba": 112.0,
            "nfl": 23.0,
            "mlb": 4.5,
            "nhl": 3.0,
            "soccer": 2.7,
        }
        return averages.get(sport.lower(), 100.0)


# Global service instance
prediction_service = PredictionService()
