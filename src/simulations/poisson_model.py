"""
Poisson distribution model for predicting game scores.
Best suited for low-scoring sports (soccer, hockey).
"""

import numpy as np
from scipy.stats import poisson
from typing import Dict, Tuple, Optional
from loguru import logger


class PoissonModel:
    """
    Poisson-based prediction model for sports.

    Uses historical scoring rates to predict game outcomes
    based on team offensive and defensive strengths.
    """

    def __init__(self):
        """Initialize Poisson model."""
        self.league_avg_goals = None
        self.team_stats = {}

    def calculate_team_strengths(
        self,
        team_id: str,
        goals_scored: float,
        goals_conceded: float,
        games_played: int,
        league_avg: float
    ) -> Dict[str, float]:
        """
        Calculate team's offensive and defensive strengths.

        Args:
            team_id: Team identifier
            goals_scored: Total goals scored
            goals_conceded: Total goals conceded
            games_played: Number of games played
            league_avg: League average goals per game

        Returns:
            Dictionary with attack_strength and defense_strength
        """
        if games_played == 0:
            return {
                "attack_strength": 1.0,
                "defense_strength": 1.0,
            }

        goals_per_game = goals_scored / games_played
        goals_against_per_game = goals_conceded / games_played

        # Attack strength: how good team is at scoring vs league average
        attack_strength = goals_per_game / league_avg if league_avg > 0 else 1.0

        # Defense strength: how good team is at preventing goals
        # Lower is better, so we invert the ratio
        defense_strength = goals_against_per_game / league_avg if league_avg > 0 else 1.0

        self.team_stats[team_id] = {
            "attack_strength": attack_strength,
            "defense_strength": defense_strength,
            "goals_per_game": goals_per_game,
            "goals_against_per_game": goals_against_per_game,
        }

        return self.team_stats[team_id]

    def predict_expected_goals(
        self,
        home_attack: float,
        home_defense: float,
        away_attack: float,
        away_defense: float,
        league_avg: float,
        home_advantage: float = 1.15
    ) -> Tuple[float, float]:
        """
        Calculate expected goals for home and away teams.

        Args:
            home_attack: Home team attack strength
            home_defense: Home team defense strength
            away_attack: Away team attack strength
            away_defense: Away team defense strength
            league_avg: League average goals per game
            home_advantage: Home field advantage multiplier (default 1.15)

        Returns:
            Tuple of (expected_home_goals, expected_away_goals)
        """
        # Expected goals = Team's attack strength × Opponent's defense weakness × League avg
        expected_home = home_attack * away_defense * league_avg * home_advantage
        expected_away = away_attack * home_defense * league_avg

        logger.debug(
            f"Expected goals - Home: {expected_home:.2f}, Away: {expected_away:.2f}"
        )

        return expected_home, expected_away

    def calculate_outcome_probabilities(
        self,
        expected_home: float,
        expected_away: float,
        max_goals: int = 10
    ) -> Dict[str, float]:
        """
        Calculate probabilities for home win, draw, away win.

        Uses Poisson distribution to calculate probability of each score,
        then sums to get overall outcome probabilities.

        Args:
            expected_home: Expected goals for home team
            expected_away: Expected goals for away team
            max_goals: Maximum goals to consider (default 10)

        Returns:
            Dictionary with win/draw/loss probabilities
        """
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0

        # Calculate probability for each possible score
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                # Probability of this exact score
                prob_home_score = poisson.pmf(home_goals, expected_home)
                prob_away_score = poisson.pmf(away_goals, expected_away)
                prob_score = prob_home_score * prob_away_score

                # Add to appropriate outcome
                if home_goals > away_goals:
                    home_win_prob += prob_score
                elif home_goals == away_goals:
                    draw_prob += prob_score
                else:
                    away_win_prob += prob_score

        return {
            "home_win": home_win_prob,
            "draw": draw_prob,
            "away_win": away_win_prob,
        }

    def calculate_score_probabilities(
        self,
        expected_home: float,
        expected_away: float,
        max_goals: int = 10
    ) -> Dict[Tuple[int, int], float]:
        """
        Calculate probability distribution for all possible scores.

        Args:
            expected_home: Expected goals for home team
            expected_away: Expected goals for away team
            max_goals: Maximum goals to consider

        Returns:
            Dictionary mapping (home_goals, away_goals) to probability
        """
        score_probs = {}

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob_home = poisson.pmf(home_goals, expected_home)
                prob_away = poisson.pmf(away_goals, expected_away)
                score_probs[(home_goals, away_goals)] = prob_home * prob_away

        return score_probs

    def predict_most_likely_score(
        self,
        expected_home: float,
        expected_away: float,
        max_goals: int = 10
    ) -> Tuple[int, int, float]:
        """
        Find the most likely final score.

        Args:
            expected_home: Expected goals for home team
            expected_away: Expected goals for away team
            max_goals: Maximum goals to consider

        Returns:
            Tuple of (home_goals, away_goals, probability)
        """
        score_probs = self.calculate_score_probabilities(
            expected_home,
            expected_away,
            max_goals
        )

        # Find score with highest probability
        most_likely_score = max(score_probs.items(), key=lambda x: x[1])

        return (
            most_likely_score[0][0],  # home goals
            most_likely_score[0][1],  # away goals
            most_likely_score[1],     # probability
        )

    def predict_over_under(
        self,
        expected_home: float,
        expected_away: float,
        total_line: float,
        max_goals: int = 15
    ) -> Dict[str, float]:
        """
        Calculate probability of over/under for a total goals line.

        Args:
            expected_home: Expected goals for home team
            expected_away: Expected goals for away team
            total_line: Total goals line (e.g., 2.5, 3.5)
            max_goals: Maximum total goals to consider

        Returns:
            Dictionary with over and under probabilities
        """
        over_prob = 0.0
        under_prob = 0.0

        score_probs = self.calculate_score_probabilities(
            expected_home,
            expected_away,
            max_goals
        )

        for (home_goals, away_goals), prob in score_probs.items():
            total_goals = home_goals + away_goals

            if total_goals > total_line:
                over_prob += prob
            else:
                under_prob += prob

        return {
            "over": over_prob,
            "under": under_prob,
        }

    def predict_game(
        self,
        home_team_stats: Dict[str, float],
        away_team_stats: Dict[str, float],
        league_avg: float,
        home_advantage: float = 1.15
    ) -> Dict[str, any]:
        """
        Full game prediction using Poisson model.

        Args:
            home_team_stats: Dict with 'attack_strength' and 'defense_strength'
            away_team_stats: Dict with 'attack_strength' and 'defense_strength'
            league_avg: League average goals per game
            home_advantage: Home field advantage multiplier

        Returns:
            Comprehensive prediction dictionary
        """
        # Calculate expected goals
        expected_home, expected_away = self.predict_expected_goals(
            home_attack=home_team_stats["attack_strength"],
            home_defense=home_team_stats["defense_strength"],
            away_attack=away_team_stats["attack_strength"],
            away_defense=away_team_stats["defense_strength"],
            league_avg=league_avg,
            home_advantage=home_advantage,
        )

        # Calculate outcome probabilities
        outcomes = self.calculate_outcome_probabilities(expected_home, expected_away)

        # Find most likely score
        likely_score = self.predict_most_likely_score(expected_home, expected_away)

        # Calculate over/under for common lines
        total_expected = expected_home + expected_away
        ou_2_5 = self.predict_over_under(expected_home, expected_away, 2.5)
        ou_3_5 = self.predict_over_under(expected_home, expected_away, 3.5)

        return {
            "expected_goals": {
                "home": expected_home,
                "away": expected_away,
                "total": total_expected,
            },
            "probabilities": outcomes,
            "most_likely_score": {
                "home": likely_score[0],
                "away": likely_score[1],
                "probability": likely_score[2],
            },
            "over_under": {
                "2.5": ou_2_5,
                "3.5": ou_3_5,
            },
            "model": "poisson",
        }


def predict_soccer_game(
    home_goals_scored: float,
    home_goals_conceded: float,
    away_goals_scored: float,
    away_goals_conceded: float,
    games_played: int,
    league_avg_goals: float = 2.7,
    home_advantage: float = 1.15
) -> Dict[str, any]:
    """
    Convenience function for predicting a soccer game.

    Args:
        home_goals_scored: Home team's total goals scored this season
        home_goals_conceded: Home team's total goals conceded
        away_goals_scored: Away team's total goals scored
        away_goals_conceded: Away team's total goals conceded
        games_played: Number of games played by teams
        league_avg_goals: League average goals per game (default 2.7 for soccer)
        home_advantage: Home field multiplier

    Returns:
        Prediction dictionary
    """
    model = PoissonModel()

    # Calculate team strengths
    home_stats = model.calculate_team_strengths(
        "home",
        home_goals_scored,
        home_goals_conceded,
        games_played,
        league_avg_goals
    )

    away_stats = model.calculate_team_strengths(
        "away",
        away_goals_scored,
        away_goals_conceded,
        games_played,
        league_avg_goals
    )

    # Make prediction
    return model.predict_game(home_stats, away_stats, league_avg_goals, home_advantage)


if __name__ == "__main__":
    # Example: Predict a soccer match
    print("\n=== Poisson Model Example ===\n")

    # Team A: Strong attacking (30 goals), weak defense (20 conceded) in 15 games
    # Team B: Balanced (22 goals, 18 conceded) in 15 games
    # League average: 2.7 goals per game

    prediction = predict_soccer_game(
        home_goals_scored=30,
        home_goals_conceded=20,
        away_goals_scored=22,
        away_goals_conceded=18,
        games_played=15,
        league_avg_goals=2.7,
        home_advantage=1.15
    )

    print("Expected Goals:")
    print(f"  Home: {prediction['expected_goals']['home']:.2f}")
    print(f"  Away: {prediction['expected_goals']['away']:.2f}")
    print(f"  Total: {prediction['expected_goals']['total']:.2f}")

    print("\nOutcome Probabilities:")
    print(f"  Home Win: {prediction['probabilities']['home_win']:.1%}")
    print(f"  Draw: {prediction['probabilities']['draw']:.1%}")
    print(f"  Away Win: {prediction['probabilities']['away_win']:.1%}")

    print("\nMost Likely Score:")
    print(f"  {prediction['most_likely_score']['home']}-"
          f"{prediction['most_likely_score']['away']} "
          f"({prediction['most_likely_score']['probability']:.1%} chance)")

    print("\nOver/Under 2.5:")
    print(f"  Over: {prediction['over_under']['2.5']['over']:.1%}")
    print(f"  Under: {prediction['over_under']['2.5']['under']:.1%}")
