"""
Pythagorean Expectations Model for Win Probability
Based on Bill James' baseball formula, adapted for multiple sports
Calibrated exponents based on commercial research
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TeamStats:
    """Season statistics for Pythagorean calculation"""
    points_for: float
    points_against: float
    games_played: int
    wins: int
    losses: int
    recent_ppg: Optional[float] = None  # Recent games scoring average
    recent_papg: Optional[float] = None  # Recent games points against
    home_ppg: Optional[float] = None
    away_ppg: Optional[float] = None
    home_papg: Optional[float] = None
    away_papg: Optional[float] = None


class PythagoreanExpectations:
    """
    Pythagorean Expectations model for win probability estimation
    Uses points scored and allowed to predict expected win percentage
    """

    # Sport-specific exponents based on empirical research
    SPORT_EXPONENTS = {
        'NBA': 13.91,    # Higher exponent due to consistent scoring
        'NFL': 2.37,     # Lower due to higher variance
        'MLB': 1.83,     # Classic Pythagorean exponent
        'NHL': 2.15,     # Goals are rarer
        'Soccer': 1.35,  # Very low scoring
        'NCAAB': 11.5,   # More variance than NBA
        'NCAAF': 2.5,    # More variance than NFL
    }

    # Adjustment factors for current season progress
    SEASON_PROGRESS_WEIGHTS = {
        'early': 0.3,   # < 20% of season
        'mid': 0.6,     # 20-60% of season
        'late': 0.8,    # 60-80% of season
        'final': 1.0    # > 80% of season
    }

    def __init__(self, sport: str = 'NBA', custom_exponent: Optional[float] = None):
        """
        Initialize Pythagorean Expectations model

        Args:
            sport: Sport type for exponent selection
            custom_exponent: Optional custom exponent override
        """
        self.sport = sport.upper()

        if custom_exponent:
            self.exponent = custom_exponent
        else:
            self.exponent = self.SPORT_EXPONENTS.get(self.sport, 2.0)

        logger.info(f"Initialized Pythagorean model for {sport} with exponent {self.exponent}")

    def calculate_expected_win_pct(self,
                                  points_for: float,
                                  points_against: float) -> float:
        """
        Calculate expected win percentage using Pythagorean formula

        Formula: Win% = PF^exp / (PF^exp + PA^exp)

        Args:
            points_for: Total points scored
            points_against: Total points allowed

        Returns:
            Expected win percentage (0-1)
        """
        if points_for <= 0 or points_against <= 0:
            return 0.5  # Default to 50% if no data

        pf_exp = points_for ** self.exponent
        pa_exp = points_against ** self.exponent

        expected_win_pct = pf_exp / (pf_exp + pa_exp)

        return expected_win_pct

    def calculate_calibrated_exponent(self,
                                     historical_data: List[TeamStats]) -> float:
        """
        Calibrate the exponent based on historical data
        Minimizes the difference between predicted and actual win percentages

        Args:
            historical_data: List of team season statistics

        Returns:
            Optimal exponent for the data
        """
        if not historical_data:
            return self.exponent

        best_exponent = self.exponent
        best_error = float('inf')

        # Try different exponents
        for exp in np.arange(0.5, 20, 0.1):
            total_error = 0

            for team in historical_data:
                if team.games_played == 0:
                    continue

                # Calculate expected win percentage with this exponent
                pf_exp = team.points_for ** exp
                pa_exp = team.points_against ** exp
                expected_win_pct = pf_exp / (pf_exp + pa_exp)

                # Calculate actual win percentage
                actual_win_pct = team.wins / team.games_played

                # Calculate error
                error = (expected_win_pct - actual_win_pct) ** 2
                total_error += error

            # Check if this is the best exponent
            avg_error = total_error / len(historical_data)
            if avg_error < best_error:
                best_error = avg_error
                best_exponent = exp

        logger.info(f"Calibrated exponent: {best_exponent:.2f} (MSE: {best_error:.4f})")
        return best_exponent

    def predict_game_probability(self,
                                home_stats: TeamStats,
                                away_stats: TeamStats,
                                use_recent: bool = True,
                                neutral_site: bool = False) -> Dict[str, float]:
        """
        Predict game outcome using Pythagorean expectations

        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            use_recent: Whether to use recent form instead of season totals
            neutral_site: Whether game is at neutral venue

        Returns:
            Dictionary with prediction details
        """
        # Determine which stats to use
        if use_recent and home_stats.recent_ppg and away_stats.recent_ppg:
            # Use recent form
            home_ppg = home_stats.recent_ppg
            home_papg = home_stats.recent_papg or home_stats.points_against / max(home_stats.games_played, 1)
            away_ppg = away_stats.recent_ppg
            away_papg = away_stats.recent_papg or away_stats.points_against / max(away_stats.games_played, 1)
        elif not neutral_site and home_stats.home_ppg and away_stats.away_ppg:
            # Use venue-specific stats
            home_ppg = home_stats.home_ppg
            home_papg = home_stats.home_papg or home_stats.points_against / max(home_stats.games_played, 1)
            away_ppg = away_stats.away_ppg
            away_papg = away_stats.away_papg or away_stats.points_against / max(away_stats.games_played, 1)
        else:
            # Use season totals
            home_ppg = home_stats.points_for / max(home_stats.games_played, 1)
            home_papg = home_stats.points_against / max(home_stats.games_played, 1)
            away_ppg = away_stats.points_for / max(away_stats.games_played, 1)
            away_papg = away_stats.points_against / max(away_stats.games_played, 1)

        # Calculate expected scores
        # Home expected = (Home offense + Away defense) / 2
        # Away expected = (Away offense + Home defense) / 2
        home_expected = (home_ppg + away_papg) / 2
        away_expected = (away_ppg + home_papg) / 2

        # Apply home advantage if not neutral site
        if not neutral_site:
            home_factor = self._get_home_advantage_factor()
            home_expected *= home_factor
            away_expected /= home_factor

        # Calculate win probabilities using log5 method
        home_pythag = self.calculate_expected_win_pct(home_ppg, home_papg)
        away_pythag = self.calculate_expected_win_pct(away_ppg, away_papg)

        # Log5 formula for head-to-head probability
        home_win_prob = self._log5_probability(home_pythag, away_pythag)

        # Adjust for season progress (less reliable early in season)
        season_progress = self._get_season_progress_weight(home_stats, away_stats)
        adjusted_home_prob = home_win_prob * season_progress + 0.5 * (1 - season_progress)

        # Calculate expected spread
        expected_spread = self._calculate_spread(home_expected, away_expected)

        # Calculate confidence based on games played and consistency
        confidence = self._calculate_confidence(home_stats, away_stats)

        return {
            'home_win_probability': adjusted_home_prob,
            'away_win_probability': 1 - adjusted_home_prob,
            'home_pythagorean': home_pythag,
            'away_pythagorean': away_pythag,
            'expected_home_score': home_expected,
            'expected_away_score': away_expected,
            'expected_spread': expected_spread,
            'confidence': confidence,
            'season_weight': season_progress,
            'model': 'pythagorean_expectations'
        }

    def _log5_probability(self, prob_a: float, prob_b: float) -> float:
        """
        Calculate head-to-head probability using Bill James' log5 formula

        Formula: P(A beats B) = (A * (1-B)) / (A * (1-B) + B * (1-A))

        Args:
            prob_a: Team A's winning percentage
            prob_b: Team B's winning percentage

        Returns:
            Probability that team A beats team B
        """
        if prob_a == 0 and prob_b == 0:
            return 0.5
        if prob_a == 1:
            return 1.0
        if prob_b == 1:
            return 0.0

        numerator = prob_a * (1 - prob_b)
        denominator = prob_a * (1 - prob_b) + prob_b * (1 - prob_a)

        if denominator == 0:
            return 0.5

        return numerator / denominator

    def _get_home_advantage_factor(self) -> float:
        """Get sport-specific home advantage multiplier"""
        home_advantages = {
            'NBA': 1.03,     # 3% scoring boost
            'NFL': 1.025,    # 2.5% boost
            'MLB': 1.01,     # 1% boost
            'NHL': 1.02,     # 2% boost
            'Soccer': 1.04,  # 4% boost (strongest)
            'NCAAB': 1.035,  # 3.5% boost
            'NCAAF': 1.03    # 3% boost
        }
        return home_advantages.get(self.sport, 1.02)

    def _get_season_progress_weight(self,
                                   home_stats: TeamStats,
                                   away_stats: TeamStats) -> float:
        """Calculate weight based on season progress"""
        # Average games played
        avg_games = (home_stats.games_played + away_stats.games_played) / 2

        # Season lengths by sport
        season_lengths = {
            'NBA': 82,
            'NFL': 17,
            'MLB': 162,
            'NHL': 82,
            'Soccer': 38,
            'NCAAB': 31,
            'NCAAF': 12
        }

        season_length = season_lengths.get(self.sport, 30)
        progress = avg_games / season_length

        if progress < 0.2:
            return self.SEASON_PROGRESS_WEIGHTS['early']
        elif progress < 0.6:
            return self.SEASON_PROGRESS_WEIGHTS['mid']
        elif progress < 0.8:
            return self.SEASON_PROGRESS_WEIGHTS['late']
        else:
            return self.SEASON_PROGRESS_WEIGHTS['final']

    def _calculate_spread(self, home_expected: float, away_expected: float) -> float:
        """Calculate expected point spread"""
        return home_expected - away_expected

    def _calculate_confidence(self,
                            home_stats: TeamStats,
                            away_stats: TeamStats) -> float:
        """
        Calculate confidence based on sample size and consistency
        """
        # Minimum games for confidence
        min_games = 5

        # Check games played
        if home_stats.games_played < min_games or away_stats.games_played < min_games:
            return 0.3  # Low confidence

        # Calculate consistency (using coefficient of variation if we had std dev)
        # For now, use games played as proxy
        avg_games = (home_stats.games_played + away_stats.games_played) / 2

        # Season lengths
        season_lengths = {'NBA': 82, 'NFL': 17, 'MLB': 162, 'NHL': 82,
                         'Soccer': 38, 'NCAAB': 31, 'NCAAF': 12}
        season_length = season_lengths.get(self.sport, 30)

        # Confidence increases with more games
        confidence = min(0.9, 0.3 + (avg_games / season_length) * 0.6)

        return confidence

    def predict_season_wins(self,
                           team_stats: TeamStats,
                           remaining_games: int) -> Dict[str, float]:
        """
        Project total season wins based on Pythagorean expectation

        Args:
            team_stats: Current team statistics
            remaining_games: Number of games remaining

        Returns:
            Dictionary with season projections
        """
        # Calculate current Pythagorean win percentage
        pythag_pct = self.calculate_expected_win_pct(
            team_stats.points_for,
            team_stats.points_against
        )

        # Project remaining wins
        expected_remaining_wins = pythag_pct * remaining_games

        # Total projected wins
        projected_total_wins = team_stats.wins + expected_remaining_wins

        # Calculate confidence interval (simplified - assumes binomial)
        std_dev = np.sqrt(remaining_games * pythag_pct * (1 - pythag_pct))
        lower_bound = max(team_stats.wins, projected_total_wins - 2 * std_dev)
        upper_bound = min(team_stats.wins + remaining_games,
                         projected_total_wins + 2 * std_dev)

        return {
            'current_wins': team_stats.wins,
            'pythagorean_pct': pythag_pct,
            'expected_remaining_wins': expected_remaining_wins,
            'projected_total_wins': projected_total_wins,
            'confidence_interval': (lower_bound, upper_bound),
            'remaining_games': remaining_games
        }


def calculate_modified_pythagorean(points_for: float,
                                  points_against: float,
                                  sport: str = 'NBA') -> float:
    """
    Convenience function for quick Pythagorean calculation

    Args:
        points_for: Points scored
        points_against: Points allowed
        sport: Sport type

    Returns:
        Expected win percentage
    """
    model = PythagoreanExpectations(sport)
    return model.calculate_expected_win_pct(points_for, points_against)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("PYTHAGOREAN EXPECTATIONS MODEL EXAMPLE")
    print("=" * 60)

    # NBA Example
    model = PythagoreanExpectations(sport='NBA')

    # Lakers stats (example)
    lakers = TeamStats(
        points_for=2800,  # Total points scored
        points_against=2650,  # Total points allowed
        games_played=30,
        wins=20,
        losses=10,
        recent_ppg=115,  # Last 10 games
        recent_papg=108,
        home_ppg=118,
        away_ppg=110
    )

    # Celtics stats (example)
    celtics = TeamStats(
        points_for=2900,
        points_against=2700,
        games_played=31,
        wins=22,
        losses=9,
        recent_ppg=112,
        recent_papg=105,
        home_ppg=115,
        away_ppg=109
    )

    print("\nTeam Pythagorean Win Percentages:")
    lakers_pythag = model.calculate_expected_win_pct(lakers.points_for, lakers.points_against)
    celtics_pythag = model.calculate_expected_win_pct(celtics.points_for, celtics.points_against)

    print(f"  Lakers: {lakers_pythag:.1%} (Actual: {lakers.wins/lakers.games_played:.1%})")
    print(f"  Celtics: {celtics_pythag:.1%} (Actual: {celtics.wins/celtics.games_played:.1%})")

    # Game prediction
    print("\n" + "-" * 40)
    print("GAME PREDICTION: Lakers @ Celtics")
    print("-" * 40)

    prediction = model.predict_game_probability(
        home_stats=celtics,  # Celtics are home
        away_stats=lakers,
        use_recent=True,
        neutral_site=False
    )

    print(f"\nWin Probabilities:")
    print(f"  Celtics (home): {prediction['home_win_probability']:.1%}")
    print(f"  Lakers (away): {prediction['away_win_probability']:.1%}")

    print(f"\nExpected Score:")
    print(f"  Celtics: {prediction['expected_home_score']:.1f}")
    print(f"  Lakers: {prediction['expected_away_score']:.1f}")
    print(f"  Spread: {prediction['expected_spread']:+.1f}")

    print(f"\nModel Details:")
    print(f"  Celtics Pythagorean: {prediction['home_pythagorean']:.1%}")
    print(f"  Lakers Pythagorean: {prediction['away_pythagorean']:.1%}")
    print(f"  Confidence: {prediction['confidence']:.1%}")
    print(f"  Season Weight: {prediction['season_weight']:.1%}")

    # Season projection
    print("\n" + "-" * 40)
    print("SEASON PROJECTION FOR LAKERS")
    print("-" * 40)

    remaining = 82 - lakers.games_played
    projection = model.predict_season_wins(lakers, remaining)

    print(f"\nCurrent Record: {projection['current_wins']}-{lakers.losses}")
    print(f"Pythagorean Win %: {projection['pythagorean_pct']:.1%}")
    print(f"Expected Remaining Wins: {projection['expected_remaining_wins']:.1f}")
    print(f"Projected Final Record: {projection['projected_total_wins']:.1f} wins")
    print(f"95% Confidence Interval: [{projection['confidence_interval'][0]:.1f}, "
          f"{projection['confidence_interval'][1]:.1f}]")