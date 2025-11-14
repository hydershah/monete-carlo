"""
Enhanced Poisson/Skellam Model for Low-Scoring Sports
Implements advanced score differential modeling using Skellam distribution
Based on commercial best practices from doc3 research
"""

import numpy as np
from scipy.stats import poisson, skellam
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
from numba import jit

logger = logging.getLogger(__name__)


@dataclass
class TeamStrength:
    """Team offensive and defensive strength parameters"""
    attack: float  # Offensive strength multiplier
    defense: float  # Defensive strength multiplier (lower is better)
    home_attack: Optional[float] = None  # Home-specific attack
    away_attack: Optional[float] = None  # Away-specific attack
    home_defense: Optional[float] = None  # Home-specific defense
    away_defense: Optional[float] = None  # Away-specific defense


class PoissonSkellamModel:
    """
    Advanced Poisson/Skellam model for low-scoring sports predictions
    Handles soccer, hockey, and other sports where goals are rare events
    """

    # Sport-specific configurations
    SPORT_CONFIGS = {
        'soccer': {
            'league_avg_goals': 2.7,
            'home_advantage': 1.15,  # 15% home boost
            'max_goals': 10,
            'common_totals': [1.5, 2.5, 3.5, 4.5],
            'draw_weight': 1.1  # Slightly increase draw probability
        },
        'hockey': {
            'league_avg_goals': 3.0,
            'home_advantage': 1.10,  # 10% home boost
            'max_goals': 12,
            'common_totals': [4.5, 5.5, 6.5, 7.5],
            'draw_weight': 1.0,
            'empty_net_factor': 1.05  # Late game scoring increase
        },
        'mls': {
            'league_avg_goals': 2.9,
            'home_advantage': 1.18,  # Strong home advantage in MLS
            'max_goals': 10,
            'common_totals': [2.5, 3.5, 4.5],
            'draw_weight': 1.05
        },
        'epl': {
            'league_avg_goals': 2.8,
            'home_advantage': 1.12,
            'max_goals': 10,
            'common_totals': [1.5, 2.5, 3.5, 4.5],
            'draw_weight': 1.08
        }
    }

    def __init__(self, sport: str = 'soccer'):
        """
        Initialize Poisson/Skellam model

        Args:
            sport: Sport type for configuration
        """
        self.sport = sport.lower()
        self.config = self.SPORT_CONFIGS.get(self.sport, self.SPORT_CONFIGS['soccer'])

    def calculate_expected_goals(self,
                                home_strength: TeamStrength,
                                away_strength: TeamStrength,
                                adjustments: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
        """
        Calculate expected goals using team strengths

        Args:
            home_strength: Home team offensive/defensive ratings
            away_strength: Away team offensive/defensive ratings
            adjustments: Optional adjustments (weather, injuries, etc.)

        Returns:
            Tuple of (home_expected, away_expected)
        """
        league_avg = self.config['league_avg_goals']
        home_advantage = self.config['home_advantage']

        # Use venue-specific strengths if available
        home_attack = home_strength.home_attack or home_strength.attack
        home_defense = home_strength.home_defense or home_strength.defense
        away_attack = away_strength.away_attack or away_strength.attack
        away_defense = away_strength.away_defense or away_strength.defense

        # Base expected goals
        home_expected = home_attack * away_defense * league_avg * home_advantage
        away_expected = away_attack * home_defense * league_avg

        # Apply adjustments
        if adjustments:
            # Weather adjustments (rain/snow reduce scoring)
            if 'weather_factor' in adjustments:
                factor = 1 - adjustments['weather_factor']  # 0 to 1, where 1 is worst weather
                home_expected *= (1 - factor * 0.2)  # Up to 20% reduction
                away_expected *= (1 - factor * 0.2)

            # Injury adjustments
            if 'home_injury_factor' in adjustments:
                home_expected *= (1 - adjustments['home_injury_factor'])
            if 'away_injury_factor' in adjustments:
                away_expected *= (1 - adjustments['away_injury_factor'])

            # Motivation factors (derby, relegation battle, title race)
            if 'home_motivation' in adjustments:
                home_expected *= (1 + adjustments['home_motivation'] * 0.1)
            if 'away_motivation' in adjustments:
                away_expected *= (1 + adjustments['away_motivation'] * 0.1)

        # Empty net adjustment for hockey
        if self.sport == 'hockey' and adjustments and adjustments.get('late_game'):
            empty_net = self.config.get('empty_net_factor', 1.0)
            home_expected *= empty_net
            away_expected *= empty_net

        return home_expected, away_expected

    def calculate_outcome_probabilities_poisson(self,
                                               lambda_home: float,
                                               lambda_away: float) -> Dict[str, float]:
        """
        Calculate win/draw/loss probabilities using Poisson distribution

        Args:
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team

        Returns:
            Dictionary with outcome probabilities
        """
        max_goals = self.config['max_goals']
        draw_weight = self.config.get('draw_weight', 1.0)

        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        # Calculate joint probability for each scoreline
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob_h = poisson.pmf(h, lambda_home)
                prob_a = poisson.pmf(a, lambda_away)
                joint_prob = prob_h * prob_a

                if h > a:
                    home_win += joint_prob
                elif h == a:
                    # Apply draw weight adjustment (some leagues have more draws)
                    draw += joint_prob * draw_weight
                else:
                    away_win += joint_prob

        # Normalize probabilities
        total = home_win + draw + away_win
        if total > 0:
            home_win /= total
            draw /= total
            away_win /= total

        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win
        }

    def calculate_spread_probabilities_skellam(self,
                                              lambda_home: float,
                                              lambda_away: float,
                                              spread: float) -> Dict[str, float]:
        """
        Calculate spread probabilities using Skellam distribution
        Skellam models the difference between two Poisson variables

        Args:
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team
            spread: Point spread (negative = home favored)

        Returns:
            Dictionary with spread coverage probabilities
        """
        # Skellam parameters
        mu1 = lambda_home
        mu2 = lambda_away

        # Calculate probability of covering spread
        # Home covers if (home - away) > spread
        home_cover_prob = 0.0
        away_cover_prob = 0.0
        push_prob = 0.0

        # Use Skellam PMF for score differential
        for diff in range(-20, 21):  # Consider differentials from -20 to +20
            prob = skellam.pmf(diff, mu1, mu2)

            if diff > spread:
                home_cover_prob += prob
            elif diff < spread:
                away_cover_prob += prob
            else:
                push_prob += prob

        return {
            'home_cover': home_cover_prob,
            'away_cover': away_cover_prob,
            'push': push_prob
        }

    def calculate_total_probabilities(self,
                                     lambda_home: float,
                                     lambda_away: float,
                                     totals: Optional[List[float]] = None) -> Dict[float, Dict[str, float]]:
        """
        Calculate over/under probabilities for multiple totals

        Args:
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team
            totals: List of total lines to evaluate

        Returns:
            Dictionary mapping total to over/under probabilities
        """
        if totals is None:
            totals = self.config['common_totals']

        results = {}

        for total_line in totals:
            over_prob = 0.0
            under_prob = 0.0
            push_prob = 0.0

            max_total = int(total_line * 3)  # Consider up to 3x the line

            for total_goals in range(max_total + 1):
                # Probability of exactly this total
                # Sum of all combinations that give this total
                prob_total = 0.0

                for home_goals in range(min(total_goals + 1, self.config['max_goals'] + 1)):
                    away_goals = total_goals - home_goals
                    if away_goals <= self.config['max_goals']:
                        prob_h = poisson.pmf(home_goals, lambda_home)
                        prob_a = poisson.pmf(away_goals, lambda_away)
                        prob_total += prob_h * prob_a

                if total_goals > total_line:
                    over_prob += prob_total
                elif total_goals < total_line:
                    under_prob += prob_total
                else:
                    push_prob += prob_total

            results[total_line] = {
                'over': over_prob,
                'under': under_prob,
                'push': push_prob
            }

        return results

    def get_most_likely_scores(self,
                              lambda_home: float,
                              lambda_away: float,
                              top_n: int = 5) -> List[Dict]:
        """
        Get the most likely scorelines

        Args:
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team
            top_n: Number of top scores to return

        Returns:
            List of most likely scores with probabilities
        """
        score_probs = {}

        for h in range(self.config['max_goals'] + 1):
            for a in range(self.config['max_goals'] + 1):
                prob_h = poisson.pmf(h, lambda_home)
                prob_a = poisson.pmf(a, lambda_away)
                score_probs[(h, a)] = prob_h * prob_a

        # Sort by probability
        sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)

        results = []
        for (h, a), prob in sorted_scores[:top_n]:
            results.append({
                'home_score': h,
                'away_score': a,
                'probability': prob,
                'scoreline': f"{h}-{a}"
            })

        return results

    def calculate_btts_probability(self,
                                  lambda_home: float,
                                  lambda_away: float) -> Dict[str, float]:
        """
        Calculate Both Teams To Score (BTTS) probability

        Args:
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team

        Returns:
            Dictionary with BTTS yes/no probabilities
        """
        # Probability of home scoring at least 1
        prob_home_scores = 1 - poisson.pmf(0, lambda_home)

        # Probability of away scoring at least 1
        prob_away_scores = 1 - poisson.pmf(0, lambda_away)

        # Both teams score
        btts_yes = prob_home_scores * prob_away_scores

        # At least one team doesn't score
        btts_no = 1 - btts_yes

        return {
            'btts_yes': btts_yes,
            'btts_no': btts_no,
            'home_clean_sheet': poisson.pmf(0, lambda_away),
            'away_clean_sheet': poisson.pmf(0, lambda_home)
        }

    def calculate_asian_handicap(self,
                                lambda_home: float,
                                lambda_away: float,
                                handicaps: List[float] = [-0.5, 0, 0.5, 1.0]) -> Dict[float, Dict]:
        """
        Calculate Asian Handicap probabilities
        Handles quarter, half, and full handicaps

        Args:
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team
            handicaps: List of handicap values

        Returns:
            Dictionary with handicap probabilities
        """
        results = {}

        for handicap in handicaps:
            # Use Skellam for handicap calculation
            probs = self.calculate_spread_probabilities_skellam(
                lambda_home, lambda_away, -handicap  # Negative because handicap convention
            )

            # Handle quarter handicaps (split stakes)
            if handicap % 0.5 == 0.25:  # Quarter handicap
                lower = handicap - 0.25
                upper = handicap + 0.25

                lower_probs = self.calculate_spread_probabilities_skellam(
                    lambda_home, lambda_away, -lower
                )
                upper_probs = self.calculate_spread_probabilities_skellam(
                    lambda_home, lambda_away, -upper
                )

                # Average the two for quarter handicaps
                results[handicap] = {
                    'home_cover': (lower_probs['home_cover'] + upper_probs['home_cover']) / 2,
                    'away_cover': (lower_probs['away_cover'] + upper_probs['away_cover']) / 2,
                    'push': (lower_probs['push'] + upper_probs['push']) / 2
                }
            else:
                results[handicap] = probs

        return results

    def predict_full_game(self,
                         home_strength: TeamStrength,
                         away_strength: TeamStrength,
                         spread: float = 0,
                         total: float = 2.5,
                         adjustments: Optional[Dict] = None) -> Dict:
        """
        Complete game prediction with all betting markets

        Args:
            home_strength: Home team strength parameters
            away_strength: Away team strength parameters
            spread: Point spread for ATS calculations
            total: Total line for O/U calculations
            adjustments: Optional adjustments dictionary

        Returns:
            Comprehensive prediction results
        """
        # Calculate expected goals
        lambda_home, lambda_away = self.calculate_expected_goals(
            home_strength, away_strength, adjustments
        )

        # Outcome probabilities
        outcomes = self.calculate_outcome_probabilities_poisson(
            lambda_home, lambda_away
        )

        # Spread probabilities using Skellam
        spread_probs = self.calculate_spread_probabilities_skellam(
            lambda_home, lambda_away, spread
        )

        # Total probabilities
        total_probs = self.calculate_total_probabilities(
            lambda_home, lambda_away, [total]
        )[total]

        # Most likely scores
        likely_scores = self.get_most_likely_scores(lambda_home, lambda_away, top_n=5)

        # BTTS probability
        btts = self.calculate_btts_probability(lambda_home, lambda_away)

        # Asian Handicaps
        asian_handicaps = self.calculate_asian_handicap(
            lambda_home, lambda_away,
            [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
        )

        # Exact score probabilities for top scores
        exact_scores = {}
        for score in likely_scores:
            key = f"{score['home_score']}-{score['away_score']}"
            exact_scores[key] = score['probability']

        return {
            'expected_goals': {
                'home': lambda_home,
                'away': lambda_away,
                'total': lambda_home + lambda_away
            },
            'win_probabilities': outcomes,
            'spread': {
                'line': spread,
                'probabilities': spread_probs
            },
            'total': {
                'line': total,
                'probabilities': total_probs
            },
            'likely_scores': likely_scores,
            'btts': btts,
            'asian_handicaps': asian_handicaps,
            'exact_scores': exact_scores,
            'confidence': self._calculate_confidence(lambda_home, lambda_away),
            'model': 'poisson_skellam'
        }

    def _calculate_confidence(self, lambda_home: float, lambda_away: float) -> float:
        """
        Calculate model confidence based on expected goal differential
        """
        diff = abs(lambda_home - lambda_away)

        if diff > 1.0:  # Clear favorite
            return 0.80
        elif diff > 0.5:  # Moderate favorite
            return 0.70
        elif diff > 0.25:  # Slight favorite
            return 0.60
        else:  # Toss-up
            return 0.55


# Numba-optimized functions for performance
@jit(nopython=True)
def fast_poisson_matrix(lambda_home: float, lambda_away: float, max_goals: int = 10):
    """
    Fast computation of joint probability matrix
    """
    matrix = np.zeros((max_goals + 1, max_goals + 1))

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            # Poisson PMF calculation
            prob_h = np.exp(-lambda_home) * (lambda_home ** h) / np.math.factorial(h)
            prob_a = np.exp(-lambda_away) * (lambda_away ** a) / np.math.factorial(a)
            matrix[h, a] = prob_h * prob_a

    return matrix


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("POISSON/SKELLAM MODEL EXAMPLE")
    print("=" * 60)

    model = PoissonSkellamModel(sport='soccer')

    # Liverpool (strong attack) vs Manchester United (strong defense)
    liverpool = TeamStrength(
        attack=1.4,   # 40% better than average at scoring
        defense=0.7   # 30% better than average at defending
    )

    man_utd = TeamStrength(
        attack=1.1,   # 10% better than average at scoring
        defense=0.8   # 20% better than average at defending
    )

    # Make prediction
    prediction = model.predict_full_game(
        home_strength=liverpool,
        away_strength=man_utd,
        spread=-0.5,  # Liverpool favored by 0.5 goals
        total=2.5,
        adjustments={
            'home_motivation': 0.5,  # Derby match
            'weather_factor': 0.2     # Light rain
        }
    )

    print("\nExpected Goals:")
    print(f"  Liverpool: {prediction['expected_goals']['home']:.2f}")
    print(f"  Man United: {prediction['expected_goals']['away']:.2f}")

    print("\nWin Probabilities:")
    print(f"  Liverpool: {prediction['win_probabilities']['home_win']:.1%}")
    print(f"  Draw: {prediction['win_probabilities']['draw']:.1%}")
    print(f"  Man United: {prediction['win_probabilities']['away_win']:.1%}")

    print("\nSpread (-0.5 Liverpool):")
    print(f"  Liverpool covers: {prediction['spread']['probabilities']['home_cover']:.1%}")
    print(f"  Man United covers: {prediction['spread']['probabilities']['away_cover']:.1%}")

    print("\nTotal (2.5 goals):")
    print(f"  Over: {prediction['total']['probabilities']['over']:.1%}")
    print(f"  Under: {prediction['total']['probabilities']['under']:.1%}")

    print("\nMost Likely Scores:")
    for score in prediction['likely_scores'][:3]:
        print(f"  {score['scoreline']}: {score['probability']:.1%}")

    print("\nBoth Teams to Score:")
    print(f"  Yes: {prediction['btts']['btts_yes']:.1%}")
    print(f"  No: {prediction['btts']['btts_no']:.1%}")

    print(f"\nModel Confidence: {prediction['confidence']:.1%}")