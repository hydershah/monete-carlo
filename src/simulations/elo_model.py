"""
Elo rating system for sports predictions.
Works well for all sports with head-to-head matchups.
"""

import math
from typing import Dict, Tuple, Optional
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session

from ..models import Team, EloRating, Game


class EloModel:
    """
    Elo rating system for sports predictions.

    Tracks team strength over time and predicts game outcomes
    based on rating differences.
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        home_advantage: float = 100.0,
        initial_rating: float = 1500.0,
        margin_of_victory_multiplier: float = 1.0
    ):
        """
        Initialize Elo model.

        Args:
            k_factor: How quickly ratings adjust (higher = more volatile)
            home_advantage: Elo points added for home team
            initial_rating: Starting rating for new teams
            margin_of_victory_multiplier: Factor for MOV adjustment
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.mov_multiplier = margin_of_victory_multiplier

    def expected_score(
        self,
        rating_a: float,
        rating_b: float
    ) -> float:
        """
        Calculate expected win probability for team A vs team B.

        Uses the standard Elo formula:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))

        Args:
            rating_a: Elo rating of team A
            rating_b: Elo rating of team B

        Returns:
            Expected win probability for team A (0 to 1)
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def margin_of_victory_multiplier(
        self,
        score_diff: int,
        elo_diff: float
    ) -> float:
        """
        Calculate margin of victory adjustment.

        Larger wins are worth more, especially against strong opponents.

        Args:
            score_diff: Point differential (winner - loser)
            elo_diff: Elo rating difference (winner - loser)

        Returns:
            Multiplier for K-factor
        """
        if score_diff == 0:
            return 1.0

        # Base MOV multiplier
        mov_mult = math.log(abs(score_diff) + 1) * self.mov_multiplier

        # Adjust for strength of opponent
        # Beating strong opponent by a lot is worth more
        if elo_diff > 0:  # Favorite won
            autocorr = 2.2 / ((elo_diff * 0.001) + 2.2)
        else:  # Underdog won
            autocorr = 2.2 / ((-elo_diff * 0.001) + 2.2)

        return mov_mult * autocorr

    def update_ratings(
        self,
        rating_a: float,
        rating_b: float,
        score_a: int,
        score_b: int,
        is_a_home: bool = False
    ) -> Tuple[float, float]:
        """
        Update Elo ratings based on game result.

        Args:
            rating_a: Current Elo rating for team A
            rating_b: Current Elo rating for team B
            score_a: Final score for team A
            score_b: Final score for team B
            is_a_home: Whether team A is home team

        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        # Adjust for home advantage
        effective_rating_a = rating_a + (self.home_advantage if is_a_home else 0)
        effective_rating_b = rating_b + (0 if is_a_home else self.home_advantage)

        # Calculate expected scores
        expected_a = self.expected_score(effective_rating_a, effective_rating_b)
        expected_b = 1.0 - expected_a

        # Actual scores (1 for win, 0.5 for draw, 0 for loss)
        if score_a > score_b:
            actual_a = 1.0
            actual_b = 0.0
        elif score_a < score_b:
            actual_a = 0.0
            actual_b = 1.0
        else:
            actual_a = 0.5
            actual_b = 0.5

        # Calculate margin of victory multiplier
        score_diff = abs(score_a - score_b)
        winner_rating = rating_a if score_a > score_b else rating_b
        loser_rating = rating_b if score_a > score_b else rating_a
        mov_mult = self.margin_of_victory_multiplier(
            score_diff,
            winner_rating - loser_rating
        )

        # Update ratings
        k = self.k_factor * mov_mult
        new_rating_a = rating_a + k * (actual_a - expected_a)
        new_rating_b = rating_b + k * (actual_b - expected_b)

        logger.debug(
            f"Elo update: Team A {rating_a:.0f} -> {new_rating_a:.0f}, "
            f"Team B {rating_b:.0f} -> {new_rating_b:.0f}"
        )

        return new_rating_a, new_rating_b

    def predict_game(
        self,
        home_rating: float,
        away_rating: float
    ) -> Dict[str, float]:
        """
        Predict game outcome based on Elo ratings.

        Args:
            home_rating: Home team's Elo rating
            away_rating: Away team's Elo rating

        Returns:
            Dictionary with win probabilities and spread
        """
        # Adjust for home advantage
        effective_home = home_rating + self.home_advantage

        # Calculate win probabilities
        home_win_prob = self.expected_score(effective_home, away_rating)
        away_win_prob = 1.0 - home_win_prob

        # Estimate point spread (roughly 25 Elo points = 1 point spread)
        elo_diff = effective_home - away_rating
        estimated_spread = elo_diff / 25.0

        return {
            "home_win_probability": home_win_prob,
            "away_win_probability": away_win_prob,
            "estimated_spread": estimated_spread,  # Positive = home favored
            "elo_difference": elo_diff,
            "model": "elo",
        }

    def get_or_create_rating(
        self,
        team_id: int,
        sport: str,
        db: Session,
        season: Optional[str] = None
    ) -> float:
        """
        Get team's current Elo rating or create initial rating.

        Args:
            team_id: Team ID in database
            sport: Sport identifier
            db: Database session
            season: Season identifier (optional)

        Returns:
            Current Elo rating
        """
        # Get most recent rating for this team
        latest_rating = db.query(EloRating).filter(
            EloRating.team_id == team_id,
            EloRating.sport == sport.lower()
        ).order_by(EloRating.rating_date.desc()).first()

        if latest_rating:
            return latest_rating.rating
        else:
            # Create initial rating
            new_rating = EloRating(
                team_id=team_id,
                sport=sport.lower(),
                rating=self.initial_rating,
                rating_date=datetime.now(),
                season=season,
                games_played=0,
            )
            db.add(new_rating)
            db.flush()
            return self.initial_rating

    def update_rating_in_db(
        self,
        team_id: int,
        sport: str,
        new_rating: float,
        db: Session,
        season: Optional[str] = None
    ):
        """
        Save new Elo rating to database.

        Args:
            team_id: Team ID
            sport: Sport identifier
            new_rating: New Elo rating value
            db: Database session
            season: Season identifier
        """
        # Get current games played
        latest_rating = db.query(EloRating).filter(
            EloRating.team_id == team_id,
            EloRating.sport == sport.lower()
        ).order_by(EloRating.rating_date.desc()).first()

        games_played = latest_rating.games_played + 1 if latest_rating else 1

        # Create new rating entry
        rating_entry = EloRating(
            team_id=team_id,
            sport=sport.lower(),
            rating=new_rating,
            rating_date=datetime.now(),
            season=season,
            games_played=games_played,
        )
        db.add(rating_entry)

    def process_game(
        self,
        game: Game,
        db: Session,
        season: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Process a completed game and update Elo ratings.

        Args:
            game: Game object (must be completed)
            db: Database session
            season: Season identifier

        Returns:
            Dictionary with old and new ratings
        """
        if not game.home_score or not game.away_score:
            raise ValueError("Game must have final scores")

        # Get current ratings
        home_rating = self.get_or_create_rating(
            game.home_team_id,
            game.sport,
            db,
            season
        )
        away_rating = self.get_or_create_rating(
            game.away_team_id,
            game.sport,
            db,
            season
        )

        # Update ratings
        new_home, new_away = self.update_ratings(
            home_rating,
            away_rating,
            game.home_score,
            game.away_score,
            is_a_home=True
        )

        # Save to database
        self.update_rating_in_db(game.home_team_id, game.sport, new_home, db, season)
        self.update_rating_in_db(game.away_team_id, game.sport, new_away, db, season)

        return {
            "home_old": home_rating,
            "home_new": new_home,
            "away_old": away_rating,
            "away_new": new_away,
        }


# Sport-specific Elo configurations
ELO_CONFIGS = {
    "nba": {
        "k_factor": 20.0,
        "home_advantage": 100.0,
        "initial_rating": 1500.0,
        "mov_multiplier": 1.0,
    },
    "nfl": {
        "k_factor": 24.0,
        "home_advantage": 65.0,
        "initial_rating": 1500.0,
        "mov_multiplier": 1.2,
    },
    "mlb": {
        "k_factor": 10.0,
        "home_advantage": 24.0,
        "initial_rating": 1500.0,
        "mov_multiplier": 0.8,
    },
    "nhl": {
        "k_factor": 16.0,
        "home_advantage": 55.0,
        "initial_rating": 1500.0,
        "mov_multiplier": 1.0,
    },
    "soccer": {
        "k_factor": 30.0,
        "home_advantage": 100.0,
        "initial_rating": 1500.0,
        "mov_multiplier": 1.5,
    },
}


def get_elo_model(sport: str) -> EloModel:
    """
    Get Elo model configured for specific sport.

    Args:
        sport: Sport identifier

    Returns:
        Configured EloModel instance
    """
    config = ELO_CONFIGS.get(sport.lower(), ELO_CONFIGS["nba"])
    return EloModel(**config)


if __name__ == "__main__":
    # Example: NBA game prediction
    print("\n=== Elo Model Example ===\n")

    # Lakers (1600 Elo) vs Celtics (1550 Elo)
    model = get_elo_model("nba")

    print("Before game:")
    print("  Lakers Elo: 1600")
    print("  Celtics Elo: 1550")

    # Predict game
    prediction = model.predict_game(home_rating=1600, away_rating=1550)

    print("\nPrediction:")
    print(f"  Lakers (home) win probability: {prediction['home_win_probability']:.1%}")
    print(f"  Celtics (away) win probability: {prediction['away_win_probability']:.1%}")
    print(f"  Estimated spread: {prediction['estimated_spread']:.1f} points")

    # Simulate game result: Lakers win 108-102
    print("\nGame Result: Lakers 108, Celtics 102")

    new_lakers, new_celtics = model.update_ratings(
        rating_a=1600,
        rating_b=1550,
        score_a=108,
        score_b=102,
        is_a_home=True
    )

    print("\nAfter game:")
    print(f"  Lakers Elo: {new_lakers:.0f} ({new_lakers - 1600:+.0f})")
    print(f"  Celtics Elo: {new_celtics:.0f} ({new_celtics - 1550:+.0f})")
