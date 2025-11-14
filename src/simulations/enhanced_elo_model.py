"""
Enhanced Elo Rating System with Commercial Best Practices
Based on doc3 research and professional implementations
Includes sport-specific K-factors, margin of victory, regression to mean
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from sqlalchemy.orm import Session

from ..models import Team, EloRating, Game

logger = logging.getLogger(__name__)


@dataclass
class EloConfig:
    """Sport-specific Elo configuration"""
    k_factor: float
    home_advantage: float
    initial_rating: float
    mov_multiplier: float
    regression_weight: float  # How much to regress to mean between seasons
    points_per_elo: float    # Conversion factor for spread calculation
    max_mov_bonus: float     # Maximum margin of victory multiplier
    autocorr_factor: float   # Autocorrelation dampening factor


class EnhancedEloModel:
    """
    Professional-grade Elo rating system with advanced adjustments
    Based on commercial best practices from billion-dollar syndicates
    """

    # Sport-specific configurations based on doc3 research
    SPORT_CONFIGS = {
        'NBA': EloConfig(
            k_factor=20.0,          # Lower than NFL due to more games
            home_advantage=100.0,    # ~2.5-3.5 point home advantage
            initial_rating=1500.0,
            mov_multiplier=1.0,
            regression_weight=0.25,  # 25% regression between seasons
            points_per_elo=25.0,     # 25 Elo = 1 point spread
            max_mov_bonus=3.0,       # Cap MOV multiplier at 3x
            autocorr_factor=2.2
        ),
        'NFL': EloConfig(
            k_factor=32.0,          # Higher due to fewer games
            home_advantage=65.0,     # ~2.5 point home advantage
            initial_rating=1500.0,
            mov_multiplier=1.2,
            regression_weight=0.33,  # More regression due to roster turnover
            points_per_elo=25.0,
            max_mov_bonus=3.5,
            autocorr_factor=2.5
        ),
        'MLB': EloConfig(
            k_factor=15.0,          # Very low due to 162 games
            home_advantage=24.0,     # ~54% home win rate
            initial_rating=1500.0,
            mov_multiplier=0.8,      # Less emphasis on MOV in baseball
            regression_weight=0.20,
            points_per_elo=100.0,    # Different scale for runs
            max_mov_bonus=2.0,
            autocorr_factor=2.0
        ),
        'NHL': EloConfig(
            k_factor=25.0,
            home_advantage=55.0,     # ~0.3 goal advantage
            initial_rating=1500.0,
            mov_multiplier=1.0,
            regression_weight=0.25,
            points_per_elo=50.0,     # Goals are rarer
            max_mov_bonus=2.5,
            autocorr_factor=2.2
        ),
        'Soccer': EloConfig(
            k_factor=30.0,          # Matches FIFA's K-factor
            home_advantage=100.0,    # Strong home advantage (60% win rate)
            initial_rating=1500.0,
            mov_multiplier=1.5,      # Goals are valuable
            regression_weight=0.20,
            points_per_elo=100.0,
            max_mov_bonus=3.0,
            autocorr_factor=2.0
        ),
        'NCAAB': EloConfig(
            k_factor=20.0,
            home_advantage=120.0,    # Stronger home court in college
            initial_rating=1500.0,
            mov_multiplier=1.1,
            regression_weight=0.40,  # High roster turnover
            points_per_elo=25.0,
            max_mov_bonus=3.0,
            autocorr_factor=2.0
        ),
        'NCAAF': EloConfig(
            k_factor=35.0,          # High volatility in college
            home_advantage=80.0,
            initial_rating=1500.0,
            mov_multiplier=1.3,
            regression_weight=0.35,  # High roster turnover
            points_per_elo=25.0,
            max_mov_bonus=4.0,       # Blowouts more common
            autocorr_factor=2.5
        )
    }

    def __init__(self, sport: str = 'NBA', custom_config: Optional[EloConfig] = None):
        """
        Initialize Enhanced Elo model

        Args:
            sport: Sport type for configuration
            custom_config: Optional custom configuration
        """
        self.sport = sport.upper()

        if custom_config:
            self.config = custom_config
        else:
            self.config = self.SPORT_CONFIGS.get(
                self.sport,
                self.SPORT_CONFIGS['NBA']
            )

        # Track ratings history for momentum analysis
        self.ratings_history = {}

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected win probability using Elo formula
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def calculate_mov_multiplier(self,
                                score_diff: int,
                                winner_rating: float,
                                loser_rating: float) -> float:
        """
        Calculate margin of victory multiplier with autocorrelation dampening
        Based on FiveThirtyEight and professional implementations
        """
        if score_diff <= 0:
            return 1.0

        # Base MOV using logarithmic scaling (diminishing returns)
        base_mov = np.log(score_diff + 1) * self.config.mov_multiplier

        # Autocorrelation adjustment - unexpected results worth more
        elo_diff = winner_rating - loser_rating

        if elo_diff > 0:  # Favorite won
            # Reduce multiplier for expected blowouts
            autocorr = self.config.autocorr_factor / (
                (elo_diff * 0.001) + self.config.autocorr_factor
            )
        else:  # Underdog won
            # Increase multiplier for upsets
            autocorr = self.config.autocorr_factor / (
                (-elo_diff * 0.001) + self.config.autocorr_factor
            )

        # Combine and cap at maximum
        final_multiplier = min(
            base_mov * autocorr,
            self.config.max_mov_bonus
        )

        return final_multiplier

    def update_ratings(self,
                      rating_home: float,
                      rating_away: float,
                      score_home: int,
                      score_away: int,
                      neutral_site: bool = False,
                      playoff_game: bool = False) -> Tuple[float, float]:
        """
        Update Elo ratings based on game result

        Args:
            rating_home: Home team's current Elo
            rating_away: Away team's current Elo
            score_home: Home team's final score
            score_away: Away team's final score
            neutral_site: Whether game is at neutral venue
            playoff_game: Whether this is a playoff game

        Returns:
            Tuple of (new_home_rating, new_away_rating)
        """
        # Adjust for home advantage (unless neutral site)
        if neutral_site:
            home_advantage = 0
        else:
            home_advantage = self.config.home_advantage

        effective_home = rating_home + home_advantage
        effective_away = rating_away

        # Calculate expected scores
        expected_home = self.expected_score(effective_home, effective_away)
        expected_away = 1.0 - expected_home

        # Actual results (1 for win, 0 for loss, 0.5 for tie)
        if score_home > score_away:
            actual_home = 1.0
            actual_away = 0.0
            winner_rating = rating_home
            loser_rating = rating_away
        elif score_away > score_home:
            actual_home = 0.0
            actual_away = 1.0
            winner_rating = rating_away
            loser_rating = rating_home
        else:  # Tie (rare in most sports)
            actual_home = 0.5
            actual_away = 0.5
            winner_rating = max(rating_home, rating_away)
            loser_rating = min(rating_home, rating_away)

        # Calculate MOV multiplier
        score_diff = abs(score_home - score_away)
        mov_mult = self.calculate_mov_multiplier(
            score_diff, winner_rating, loser_rating
        )

        # Adjust K-factor for playoffs (more important games)
        k_factor = self.config.k_factor
        if playoff_game:
            k_factor *= 1.25  # 25% increase for playoffs

        # Apply updates
        k_adjusted = k_factor * mov_mult

        new_home = rating_home + k_adjusted * (actual_home - expected_home)
        new_away = rating_away + k_adjusted * (actual_away - expected_away)

        # Prevent ratings from going too extreme
        new_home = np.clip(new_home, 800, 2200)
        new_away = np.clip(new_away, 800, 2200)

        logger.debug(
            f"Elo update: Home {rating_home:.0f}->{new_home:.0f} "
            f"({new_home-rating_home:+.0f}), "
            f"Away {rating_away:.0f}->{new_away:.0f} "
            f"({new_away-rating_away:+.0f}), "
            f"MOV mult: {mov_mult:.2f}"
        )

        return new_home, new_away

    def regress_to_mean(self,
                       current_rating: float,
                       games_played: int = 0) -> float:
        """
        Regress ratings toward mean between seasons
        Accounts for roster turnover and uncertainty
        """
        # More games played = less regression
        game_factor = min(games_played / 20.0, 1.0)

        # Weight toward mean based on sport config
        regression_weight = self.config.regression_weight * (1 - game_factor)

        mean_rating = self.config.initial_rating
        regressed = (
            current_rating * (1 - regression_weight) +
            mean_rating * regression_weight
        )

        return regressed

    def predict_game(self,
                    home_rating: float,
                    away_rating: float,
                    neutral_site: bool = False,
                    recent_form: Optional[Dict] = None) -> Dict[str, float]:
        """
        Predict game outcome based on Elo ratings

        Args:
            home_rating: Home team's Elo rating
            away_rating: Away team's Elo rating
            neutral_site: Whether game is at neutral venue
            recent_form: Optional recent performance adjustments

        Returns:
            Dictionary with predictions
        """
        # Apply home advantage
        if neutral_site:
            effective_home = home_rating
        else:
            effective_home = home_rating + self.config.home_advantage

        # Optional recent form adjustment (momentum)
        if recent_form:
            # Note: Research shows momentum is mostly noise, but included for completeness
            home_form_adj = recent_form.get('home_momentum', 0) * 10
            away_form_adj = recent_form.get('away_momentum', 0) * 10

            effective_home += home_form_adj
            away_rating_adj = away_rating + away_form_adj
        else:
            away_rating_adj = away_rating

        # Calculate win probabilities
        home_win_prob = self.expected_score(effective_home, away_rating_adj)
        away_win_prob = 1.0 - home_win_prob

        # Calculate expected spread
        elo_diff = effective_home - away_rating_adj
        expected_spread = elo_diff / self.config.points_per_elo

        # Calculate confidence based on rating gap
        rating_gap = abs(elo_diff)
        if rating_gap > 200:
            confidence = 0.85  # High confidence
        elif rating_gap > 100:
            confidence = 0.70  # Medium confidence
        elif rating_gap > 50:
            confidence = 0.60  # Low confidence
        else:
            confidence = 0.55  # Very low confidence (toss-up)

        return {
            'home_win_probability': home_win_prob,
            'away_win_probability': away_win_prob,
            'expected_spread': expected_spread,  # Positive = home favored
            'confidence': confidence,
            'elo_difference': elo_diff,
            'home_effective_rating': effective_home,
            'away_effective_rating': away_rating_adj,
            'model': 'enhanced_elo'
        }

    def calculate_tournament_odds(self,
                                 team_ratings: Dict[str, float],
                                 tournament_structure: str = 'single_elimination',
                                 n_simulations: int = 10000) -> Dict[str, float]:
        """
        Calculate tournament/playoff odds using Elo ratings
        Uses Monte Carlo simulation for complex tournament structures
        """
        results = {team: 0 for team in team_ratings}

        for _ in range(n_simulations):
            # Simulate tournament
            remaining_teams = list(team_ratings.keys())

            while len(remaining_teams) > 1:
                next_round = []

                # Pair up teams (simplified - real implementation would use actual bracket)
                for i in range(0, len(remaining_teams) - 1, 2):
                    team1 = remaining_teams[i]
                    team2 = remaining_teams[i + 1]

                    rating1 = team_ratings[team1]
                    rating2 = team_ratings[team2]

                    # Neutral site for tournament games
                    win_prob1 = self.expected_score(rating1, rating2)

                    # Simulate game
                    if np.random.random() < win_prob1:
                        next_round.append(team1)
                    else:
                        next_round.append(team2)

                # Handle odd number of teams (bye)
                if len(remaining_teams) % 2 == 1:
                    next_round.append(remaining_teams[-1])

                remaining_teams = next_round

            # Record winner
            if remaining_teams:
                winner = remaining_teams[0]
                results[winner] += 1

        # Convert to probabilities
        for team in results:
            results[team] /= n_simulations

        return results

    def get_or_create_rating(self,
                            team_id: int,
                            sport: str,
                            db: Session,
                            season: Optional[str] = None) -> float:
        """
        Get team's current Elo rating or create initial rating
        Includes regression to mean for new seasons
        """
        # Get most recent rating
        latest_rating = db.query(EloRating).filter(
            EloRating.team_id == team_id,
            EloRating.sport == sport.lower()
        ).order_by(EloRating.rating_date.desc()).first()

        if latest_rating:
            # Check if it's a new season (> 6 months since last rating)
            days_since = (datetime.now() - latest_rating.rating_date).days

            if days_since > 180:  # New season
                # Apply regression to mean
                regressed = self.regress_to_mean(
                    latest_rating.rating,
                    latest_rating.games_played
                )

                # Create new season entry
                new_rating = EloRating(
                    team_id=team_id,
                    sport=sport.lower(),
                    rating=regressed,
                    rating_date=datetime.now(),
                    season=season,
                    games_played=0
                )
                db.add(new_rating)
                db.flush()

                return regressed
            else:
                return latest_rating.rating
        else:
            # Create initial rating
            new_rating = EloRating(
                team_id=team_id,
                sport=sport.lower(),
                rating=self.config.initial_rating,
                rating_date=datetime.now(),
                season=season,
                games_played=0
            )
            db.add(new_rating)
            db.flush()

            return self.config.initial_rating

    def batch_update_ratings(self,
                            games: List[Dict],
                            initial_ratings: Dict[str, float]) -> Dict[str, float]:
        """
        Process multiple games and update ratings
        Useful for season simulations
        """
        current_ratings = initial_ratings.copy()

        for game in games:
            home_team = game['home_team']
            away_team = game['away_team']

            # Get current ratings
            home_rating = current_ratings.get(home_team, self.config.initial_rating)
            away_rating = current_ratings.get(away_team, self.config.initial_rating)

            # Update based on result
            new_home, new_away = self.update_ratings(
                home_rating,
                away_rating,
                game['home_score'],
                game['away_score'],
                game.get('neutral_site', False),
                game.get('playoff', False)
            )

            # Store new ratings
            current_ratings[home_team] = new_home
            current_ratings[away_team] = new_away

        return current_ratings


def get_enhanced_elo_model(sport: str) -> EnhancedEloModel:
    """
    Get Enhanced Elo model configured for specific sport
    """
    return EnhancedEloModel(sport)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("ENHANCED ELO MODEL EXAMPLE")
    print("=" * 60)

    # NBA example
    model = get_enhanced_elo_model('NBA')

    print("\n1. Regular Season Game (Lakers vs Celtics)")
    print("-" * 40)
    print("Before: Lakers 1600, Celtics 1550")

    prediction = model.predict_game(
        home_rating=1600,
        away_rating=1550,
        neutral_site=False
    )

    print(f"Lakers win probability: {prediction['home_win_probability']:.1%}")
    print(f"Expected spread: {prediction['expected_spread']:.1f}")
    print(f"Confidence: {prediction['confidence']:.1%}")

    # Simulate game result
    print("\nGame Result: Lakers 108, Celtics 102 (6-point win)")

    new_lakers, new_celtics = model.update_ratings(
        rating_home=1600,
        rating_away=1550,
        score_home=108,
        score_away=102
    )

    print(f"After: Lakers {new_lakers:.0f} ({new_lakers-1600:+.0f}), "
          f"Celtics {new_celtics:.0f} ({new_celtics-1550:+.0f})")

    print("\n2. Playoff Game with Upset")
    print("-" * 40)
    print("Before: Warriors 1650, Kings 1520")

    # Underdog wins
    print("Game Result: Warriors 98, Kings 105 (Upset!)")

    new_warriors, new_kings = model.update_ratings(
        rating_home=1650,
        rating_away=1520,
        score_home=98,
        score_away=105,
        playoff_game=True
    )

    print(f"After: Warriors {new_warriors:.0f} ({new_warriors-1650:+.0f}), "
          f"Kings {new_kings:.0f} ({new_kings-1520:+.0f})")
    print("Note: Larger swing due to upset + playoff multiplier")

    print("\n3. Season Regression Example")
    print("-" * 40)
    championship_rating = 1750
    print(f"Championship team rating: {championship_rating}")

    regressed = model.regress_to_mean(championship_rating, games_played=0)
    print(f"After offseason (0 games): {regressed:.0f} ({regressed-championship_rating:+.0f})")

    regressed_10 = model.regress_to_mean(championship_rating, games_played=10)
    print(f"After 10 games into season: {regressed_10:.0f} ({regressed_10-championship_rating:+.0f})")