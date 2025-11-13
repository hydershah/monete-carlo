"""
Monte Carlo simulation engine for sports predictions.
Runs thousands of simulations to estimate probability distributions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from dataclasses import dataclass

from .poisson_model import PoissonModel
from .elo_model import EloModel


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    home_wins: int
    away_wins: int
    draws: int
    total_simulations: int
    home_scores: List[int]
    away_scores: List[int]
    score_distribution: Dict[Tuple[int, int], int]

    @property
    def home_win_probability(self) -> float:
        """Home team win probability."""
        return self.home_wins / self.total_simulations

    @property
    def away_win_probability(self) -> float:
        """Away team win probability."""
        return self.away_wins / self.total_simulations

    @property
    def draw_probability(self) -> float:
        """Draw probability."""
        return self.draws / self.total_simulations

    @property
    def average_home_score(self) -> float:
        """Average home team score."""
        return np.mean(self.home_scores)

    @property
    def average_away_score(self) -> float:
        """Average away team score."""
        return np.mean(self.away_scores)

    @property
    def average_total_score(self) -> float:
        """Average total score."""
        return self.average_home_score + self.average_away_score


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for sports predictions.

    Runs thousands of simulations using Poisson distribution
    to generate probability distributions for game outcomes.
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            n_simulations: Number of simulations to run (default 10,000)
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations

        if random_seed is not None:
            np.random.seed(random_seed)

        self.poisson_model = PoissonModel()

    def simulate_poisson_game(
        self,
        expected_home: float,
        expected_away: float
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation using Poisson distribution.

        Args:
            expected_home: Expected goals for home team
            expected_away: Expected goals for away team

        Returns:
            SimulationResult with outcomes
        """
        logger.info(
            f"Running {self.n_simulations:,} simulations with "
            f"expected goals - Home: {expected_home:.2f}, Away: {expected_away:.2f}"
        )

        # Generate random scores for all simulations at once (vectorized)
        home_scores = np.random.poisson(expected_home, self.n_simulations)
        away_scores = np.random.poisson(expected_away, self.n_simulations)

        # Count outcomes
        home_wins = np.sum(home_scores > away_scores)
        away_wins = np.sum(away_scores > home_scores)
        draws = np.sum(home_scores == away_scores)

        # Build score distribution
        score_dist = {}
        for h_score, a_score in zip(home_scores, away_scores):
            key = (int(h_score), int(a_score))
            score_dist[key] = score_dist.get(key, 0) + 1

        result = SimulationResult(
            home_wins=int(home_wins),
            away_wins=int(away_wins),
            draws=int(draws),
            total_simulations=self.n_simulations,
            home_scores=home_scores.tolist(),
            away_scores=away_scores.tolist(),
            score_distribution=score_dist,
        )

        logger.info(
            f"Simulation complete - Home: {result.home_win_probability:.1%}, "
            f"Draw: {result.draw_probability:.1%}, "
            f"Away: {result.away_win_probability:.1%}"
        )

        return result

    def simulate_with_variance(
        self,
        expected_home: float,
        expected_away: float,
        variance_factor: float = 0.1
    ) -> SimulationResult:
        """
        Run simulation with variance in expected goals.

        Adds uncertainty by varying the expected goals parameters
        in each simulation.

        Args:
            expected_home: Expected goals for home team
            expected_away: Expected goals for away team
            variance_factor: Variance as fraction of expected goals (default 0.1)

        Returns:
            SimulationResult
        """
        # Add variance to expected goals
        home_std = expected_home * variance_factor
        away_std = expected_away * variance_factor

        # Generate varying expected goals for each simulation
        varying_home = np.random.normal(expected_home, home_std, self.n_simulations)
        varying_away = np.random.normal(expected_away, away_std, self.n_simulations)

        # Ensure non-negative
        varying_home = np.maximum(varying_home, 0.1)
        varying_away = np.maximum(varying_away, 0.1)

        # Generate scores with varying parameters
        home_scores = np.array([
            np.random.poisson(lam) for lam in varying_home
        ])
        away_scores = np.array([
            np.random.poisson(lam) for lam in varying_away
        ])

        # Count outcomes
        home_wins = np.sum(home_scores > away_scores)
        away_wins = np.sum(away_scores > home_scores)
        draws = np.sum(home_scores == away_scores)

        # Build score distribution
        score_dist = {}
        for h_score, a_score in zip(home_scores, away_scores)  :
            key = (int(h_score), int(a_score))
            score_dist[key] = score_dist.get(key, 0) + 1

        return SimulationResult(
            home_wins=int(home_wins),
            away_wins=int(away_wins),
            draws=int(draws),
            total_simulations=self.n_simulations,
            home_scores=home_scores.tolist(),
            away_scores=away_scores.tolist(),
            score_distribution=score_dist,
        )

    def simulate_nba_game(
        self,
        home_ppg: float,
        away_ppg: float,
        home_defensive_rating: float = 110.0,
        away_defensive_rating: float = 110.0,
        league_avg_ppg: float = 112.0
    ) -> SimulationResult:
        """
        Simulate NBA game using adjusted scoring rates.

        Args:
            home_ppg: Home team points per game
            away_ppg: Away team points per game
            home_defensive_rating: Home defensive rating (points per 100 possessions)
            away_defensive_rating: Away defensive rating
            league_avg_ppg: League average points per game

        Returns:
            SimulationResult
        """
        # Adjust expected scores based on opponent defense
        # Using normal distribution for basketball scores
        home_std = 12.0  # Typical NBA standard deviation
        away_std = 12.0

        # Adjust for opponent defense
        home_expected = home_ppg * (league_avg_ppg / away_defensive_rating)
        away_expected = away_ppg * (league_avg_ppg / home_defensive_rating)

        # Generate scores using normal distribution (better for basketball)
        home_scores = np.random.normal(home_expected, home_std, self.n_simulations)
        away_scores = np.random.normal(away_expected, away_std, self.n_simulations)

        # Round and ensure non-negative
        home_scores = np.maximum(np.round(home_scores), 0).astype(int)
        away_scores = np.maximum(np.round(away_scores), 0).astype(int)

        # Count outcomes
        home_wins = np.sum(home_scores > away_scores)
        away_wins = np.sum(away_scores > home_scores)
        draws = np.sum(home_scores == away_scores)  # Very rare in NBA

        # Build score distribution
        score_dist = {}
        for h_score, a_score in zip(home_scores, away_scores):
            key = (int(h_score), int(a_score))
            score_dist[key] = score_dist.get(key, 0) + 1

        return SimulationResult(
            home_wins=int(home_wins),
            away_wins=int(away_wins),
            draws=int(draws),
            total_simulations=self.n_simulations,
            home_scores=home_scores.tolist(),
            away_scores=away_scores.tolist(),
            score_distribution=score_dist,
        )

    def calculate_spread_probability(
        self,
        result: SimulationResult,
        spread: float
    ) -> Dict[str, float]:
        """
        Calculate probability of covering the spread.

        Args:
            result: SimulationResult from simulation
            spread: Point spread (positive = home favored)

        Returns:
            Dict with probabilities for home_cover, push, away_cover
        """
        home_scores = np.array(result.home_scores)
        away_scores = np.array(result.away_scores)

        # Calculate margin (home - away)
        margins = home_scores - away_scores

        # Home covers if margin > spread
        home_covers = np.sum(margins > spread)
        pushes = np.sum(margins == spread)
        away_covers = np.sum(margins < spread)

        total = len(margins)

        return {
            "home_cover": home_covers / total,
            "push": pushes / total,
            "away_cover": away_covers / total,
        }

    def calculate_total_probability(
        self,
        result: SimulationResult,
        total_line: float
    ) -> Dict[str, float]:
        """
        Calculate probability of over/under.

        Args:
            result: SimulationResult from simulation
            total_line: Over/under line (e.g., 220.5 for NBA)

        Returns:
            Dict with probabilities for over, push, under
        """
        home_scores = np.array(result.home_scores)
        away_scores = np.array(result.away_scores)

        totals = home_scores + away_scores

        overs = np.sum(totals > total_line)
        pushes = np.sum(totals == total_line)
        unders = np.sum(totals < total_line)

        total_sims = len(totals)

        return {
            "over": overs / total_sims,
            "push": pushes / total_sims,
            "under": unders / total_sims,
        }

    def get_most_likely_scores(
        self,
        result: SimulationResult,
        top_n: int = 5
    ) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get the most likely final scores.

        Args:
            result: SimulationResult
            top_n: Number of top scores to return

        Returns:
            List of ((home_score, away_score), probability) tuples
        """
        # Sort by frequency
        sorted_scores = sorted(
            result.score_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Convert counts to probabilities
        top_scores = [
            (score, count / result.total_simulations)
            for score, count in sorted_scores[:top_n]
        ]

        return top_scores

    def full_game_analysis(
        self,
        expected_home: float,
        expected_away: float,
        spread: Optional[float] = None,
        total_line: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Comprehensive game analysis using Monte Carlo.

        Args:
            expected_home: Expected score for home team
            expected_away: Expected score for away team
            spread: Point spread if available
            total_line: Over/under line if available

        Returns:
            Comprehensive analysis dictionary
        """
        # Run simulation
        result = self.simulate_poisson_game(expected_home, expected_away)

        # Basic analysis
        analysis = {
            "probabilities": {
                "home_win": result.home_win_probability,
                "away_win": result.away_win_probability,
                "draw": result.draw_probability,
            },
            "expected_scores": {
                "home": result.average_home_score,
                "away": result.average_away_score,
                "total": result.average_total_score,
            },
            "most_likely_scores": self.get_most_likely_scores(result, top_n=5),
            "simulations_run": self.n_simulations,
            "model": "monte_carlo",
        }

        # Add spread analysis if provided
        if spread is not None:
            analysis["spread_analysis"] = self.calculate_spread_probability(
                result,
                spread
            )

        # Add total analysis if provided
        if total_line is not None:
            analysis["total_analysis"] = self.calculate_total_probability(
                result,
                total_line
            )

        return analysis


if __name__ == "__main__":
    # Example: NBA game simulation
    print("\n=== Monte Carlo Simulation Example ===\n")

    # Lakers vs Celtics
    # Lakers: 112 PPG, 108 defensive rating
    # Celtics: 110 PPG, 106 defensive rating

    simulator = MonteCarloSimulator(n_simulations=10000, random_seed=42)

    print("Simulating Lakers vs Celtics...")
    result = simulator.simulate_nba_game(
        home_ppg=112,
        away_ppg=110,
        home_defensive_rating=108,
        away_defensive_rating=106,
        league_avg_ppg=112
    )

    print(f"\nResults from {result.total_simulations:,} simulations:")
    print(f"  Lakers (home) win: {result.home_win_probability:.1%}")
    print(f"  Celtics (away) win: {result.away_win_probability:.1%}")
    print(f"\nExpected Scores:")
    print(f"  Lakers: {result.average_home_score:.1f}")
    print(f"  Celtics: {result.average_away_score:.1f}")
    print(f"  Total: {result.average_total_score:.1f}")

    # Spread analysis
    spread = -2.5  # Lakers favored by 2.5
    spread_prob = simulator.calculate_spread_probability(result, spread)
    print(f"\nSpread Analysis (Lakers {spread}):")
    print(f"  Lakers cover: {spread_prob['home_cover']:.1%}")
    print(f"  Celtics cover: {spread_prob['away_cover']:.1%}")

    # Total analysis
    total = 220.5
    total_prob = simulator.calculate_total_probability(result, total)
    print(f"\nOver/Under {total}:")
    print(f"  Over: {total_prob['over']:.1%}")
    print(f"  Under: {total_prob['under']:.1%}")

    # Most likely scores
    print(f"\nMost Likely Scores:")
    for (home, away), prob in simulator.get_most_likely_scores(result, top_n=3):
        print(f"  {home}-{away}: {prob:.2%}")
