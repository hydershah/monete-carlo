"""
Unified Data Manager for GameLens.ai
Provides clean interface for accessing team stats and game data
Automatically handles Redis caching and database fallbacks

Usage:
    from src.data.data_manager import DataManager

    dm = DataManager()

    # Get team stats (checks Redis first, falls back to DB)
    lakers_stats = dm.get_team_stats('lakers', 'NBA')

    # Get game data
    games = dm.get_team_games('lakers', 'NBA', days_back=30)

    # Get prediction-ready data
    prediction_data = dm.get_prediction_data('lakers', 'celtics', 'NBA')
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass

from sqlalchemy import create_engine, or_, and_
from sqlalchemy.orm import sessionmaker

from .espn_client import ESPNClient
from .goalserve_client import GoalserveClient
from ..models.database_schema import (
    GamesHistory, TeamStatsDaily, TeamRatingsHistory, Base
)
from ..simulations.pythagorean_expectations import PythagoreanExpectations
from config.redis_config import get_cache

logger = logging.getLogger(__name__)


@dataclass
class PredictionInputData:
    """
    Complete data package for making a prediction
    Contains everything needed for all 5 models
    """
    # Team identification
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    league: str

    # Current season stats
    home_stats: Dict
    away_stats: Dict

    # Ratings
    home_elo: float
    away_elo: float

    # Game context
    spread: Optional[float] = None
    total: Optional[float] = None
    neutral_site: bool = False

    # Recent form (last 10 games)
    home_recent_games: List[Dict] = None
    away_recent_games: List[Dict] = None


class DataManager:
    """
    Unified data access layer for GameLens.ai
    Handles Redis caching, database queries, and API fallbacks
    """

    def __init__(self, db_url: str = None, use_cache: bool = True):
        """
        Initialize data manager

        Args:
            db_url: PostgreSQL connection string (default from env)
            use_cache: Whether to use Redis cache (default True)
        """
        # Database connection
        self.db_url = db_url or os.getenv(
            'DATABASE_URL',
            'postgresql://user:password@localhost:5432/gamelens'
        )

        try:
            self.engine = create_engine(self.db_url, echo=False, pool_pre_ping=True)
            self.Session = sessionmaker(bind=self.engine)
            logger.info(f"✅ Connected to database")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            self.engine = None
            self.Session = None

        # Redis cache
        self.use_cache = use_cache
        self.cache = get_cache() if use_cache else None

        # API clients (for live data fallback)
        self.espn = ESPNClient()
        self.goalserve = GoalserveClient(api_key=os.getenv('GOALSERVE_API_KEY'))

    # ==================== TEAM STATS ====================

    def get_team_stats(self, team_id: str, league: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get current season stats for a team

        Strategy:
        1. Check Redis cache (fast, 6-hour TTL)
        2. Query database if cache miss
        3. Fallback to ESPN API if database empty
        4. Cache result for future requests

        Args:
            team_id: Team identifier
            league: League (NBA, NFL, etc.)
            force_refresh: Skip cache and query fresh data

        Returns:
            Dictionary of team statistics or None if not found
        """
        # Try cache first (unless forcing refresh)
        if not force_refresh and self.cache:
            cached_stats = self.cache.get_team_stats(team_id, league)
            if cached_stats:
                logger.debug(f"Cache HIT: Team stats for {team_id}")
                return cached_stats

        logger.debug(f"Cache MISS: Team stats for {team_id}")

        # Query database
        stats = self._get_team_stats_from_db(team_id, league)

        if stats:
            # Cache for future requests
            if self.cache:
                self.cache.cache_team_stats(team_id, league, stats)
            return stats

        # Fallback to live API
        logger.warning(f"No DB stats for {team_id}, fetching from ESPN...")
        stats = self._get_team_stats_from_api(team_id, league)

        if stats and self.cache:
            self.cache.cache_team_stats(team_id, league, stats)

        return stats

    def _get_team_stats_from_db(self, team_id: str, league: str) -> Optional[Dict]:
        """Query team stats from database"""
        if not self.Session:
            return None

        session = self.Session()

        try:
            # Get most recent snapshot
            snapshot = session.query(TeamStatsDaily).filter(
                TeamStatsDaily.team_id == team_id,
                TeamStatsDaily.league == league
            ).order_by(TeamStatsDaily.snapshot_date.desc()).first()

            if not snapshot:
                return None

            # Get latest Elo rating
            elo_record = session.query(TeamRatingsHistory).filter(
                TeamRatingsHistory.team_id == team_id,
                TeamRatingsHistory.league == league
            ).order_by(TeamRatingsHistory.rating_date.desc()).first()

            elo_rating = elo_record.elo_rating if elo_record else 1500

            return {
                'team_id': team_id,
                'league': league,
                'wins': snapshot.wins,
                'losses': snapshot.losses,
                'games_played': snapshot.games_played,
                'win_pct': snapshot.win_pct,
                'points_for': snapshot.points_for,
                'points_against': snapshot.points_against,
                'ppg': snapshot.ppg,
                'papg': snapshot.papg,
                'home_ppg': snapshot.home_ppg,
                'away_ppg': snapshot.away_ppg,
                'recent_ppg': snapshot.recent_ppg,
                'recent_papg': snapshot.recent_papg,
                'elo_rating': elo_rating,
                'pythagorean_win_pct': snapshot.pythagorean_win_pct,
                'last_updated': snapshot.snapshot_date.isoformat()
            }

        except Exception as e:
            logger.error(f"Error querying team stats: {e}")
            return None

        finally:
            session.close()

    def _get_team_stats_from_api(self, team_id: str, league: str) -> Optional[Dict]:
        """Fetch team stats from ESPN API"""
        try:
            # Fetch team info from ESPN
            team_info = self.espn.get_team_info(league.lower(), team_id)

            if not team_info:
                return None

            # Calculate stats from recent games
            today = date.today()
            recent_games = self.espn.get_schedule(
                league=league.lower(),
                start_date=(today - timedelta(days=90)).strftime('%Y%m%d'),
                end_date=today.strftime('%Y%m%d')
            )

            # Filter for this team
            team_games = [g for g in recent_games if
                         g.get('home_team_id') == team_id or g.get('away_team_id') == team_id]

            if not team_games:
                return None

            # Calculate stats
            wins = 0
            losses = 0
            points_for = 0
            points_against = 0

            for game in team_games:
                if game.get('status') != 'final':
                    continue

                is_home = game['home_team_id'] == team_id
                team_score = game['home_score'] if is_home else game['away_score']
                opp_score = game['away_score'] if is_home else game['home_score']

                points_for += team_score
                points_against += opp_score

                if team_score > opp_score:
                    wins += 1
                else:
                    losses += 1

            games_played = wins + losses

            if games_played == 0:
                return None

            return {
                'team_id': team_id,
                'league': league,
                'wins': wins,
                'losses': losses,
                'games_played': games_played,
                'win_pct': wins / games_played,
                'points_for': points_for,
                'points_against': points_against,
                'ppg': points_for / games_played,
                'papg': points_against / games_played,
                'elo_rating': 1500,  # Default for new teams
                'pythagorean_win_pct': self._calculate_pythagorean(points_for, points_against, league),
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching from API: {e}")
            return None

    def _calculate_pythagorean(self, points_for: float, points_against: float, league: str) -> float:
        """Calculate Pythagorean win expectation"""
        pythag = PythagoreanExpectations(sport=league)
        return pythag.calculate_expected_win_pct(points_for, points_against)

    # ==================== GAME DATA ====================

    def get_team_games(self, team_id: str, league: str, days_back: int = 30) -> List[Dict]:
        """
        Get recent games for a team

        Args:
            team_id: Team identifier
            league: League code
            days_back: Number of days to look back (default 30)

        Returns:
            List of game dictionaries
        """
        if not self.Session:
            return []

        session = self.Session()

        try:
            cutoff_date = date.today() - timedelta(days=days_back)

            games = session.query(GamesHistory).filter(
                or_(
                    GamesHistory.home_team_id == team_id,
                    GamesHistory.away_team_id == team_id
                ),
                GamesHistory.league == league,
                GamesHistory.game_date >= cutoff_date,
                GamesHistory.status == 'final'
            ).order_by(GamesHistory.game_date.desc()).all()

            return [self._game_to_dict(game, team_id) for game in games]

        except Exception as e:
            logger.error(f"Error querying games: {e}")
            return []

        finally:
            session.close()

    def _game_to_dict(self, game: GamesHistory, team_id: str) -> Dict:
        """Convert game record to dictionary"""
        is_home = game.home_team_id == team_id

        return {
            'game_id': game.game_id,
            'date': game.game_date.isoformat(),
            'opponent': game.away_team_id if is_home else game.home_team_id,
            'is_home': is_home,
            'team_score': game.home_score if is_home else game.away_score,
            'opponent_score': game.away_score if is_home else game.home_score,
            'won': (game.home_score > game.away_score) if is_home else (game.away_score > game.home_score),
            'spread': game.spread,
            'total': game.total,
            'covered': game.home_covered if is_home else (not game.home_covered if game.home_covered is not None else None),
            'went_over': game.went_over
        }

    # ==================== PREDICTION DATA ====================

    def get_prediction_data(self, home_team_id: str, away_team_id: str,
                           league: str, spread: float = None, total: float = None,
                           neutral_site: bool = False) -> Optional[PredictionInputData]:
        """
        Get complete data package for making a prediction

        This is the main method to use when making predictions
        Returns everything needed for all 5 models

        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            league: League code
            spread: Point spread (optional)
            total: Over/under total (optional)
            neutral_site: Whether game is at neutral venue

        Returns:
            PredictionInputData object with all required data
        """
        logger.info(f"Gathering prediction data: {home_team_id} vs {away_team_id} ({league})")

        # Get team stats (with caching)
        home_stats = self.get_team_stats(home_team_id, league)
        away_stats = self.get_team_stats(away_team_id, league)

        if not home_stats or not away_stats:
            logger.error(f"Could not fetch stats for teams")
            return None

        # Get recent games (last 10)
        home_recent = self.get_team_games(home_team_id, league, days_back=60)[:10]
        away_recent = self.get_team_games(away_team_id, league, days_back=60)[:10]

        # Package everything together
        return PredictionInputData(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_team_name=home_team_id,  # TODO: Get actual team names
            away_team_name=away_team_id,
            league=league,

            home_stats=home_stats,
            away_stats=away_stats,

            home_elo=home_stats.get('elo_rating', 1500),
            away_elo=away_stats.get('elo_rating', 1500),

            spread=spread,
            total=total,
            neutral_site=neutral_site,

            home_recent_games=home_recent,
            away_recent_games=away_recent
        )

    # ==================== UTILITY METHODS ====================

    def get_todays_games(self, league: str) -> List[Dict]:
        """
        Get today's scheduled games for a league

        Returns:
            List of games with basic information
        """
        try:
            # Fetch from ESPN (most reliable for current games)
            games = self.espn.get_scoreboard(
                league=league.lower(),
                date=date.today().strftime('%Y%m%d')
            )

            return games

        except Exception as e:
            logger.error(f"Error fetching today's games: {e}")
            return []

    def warmup_cache_for_games(self, games: List[Dict]):
        """
        Pre-warm cache with team stats for all teams in upcoming games
        Call this before game day to ensure fast predictions

        Args:
            games: List of game dictionaries with team IDs
        """
        logger.info("Warming up cache for upcoming games...")

        teams_processed = set()

        for game in games:
            home_team = game.get('home_team_id')
            away_team = game.get('away_team_id')
            league = game.get('league')

            for team_id in [home_team, away_team]:
                if team_id and team_id not in teams_processed:
                    # Force load into cache
                    self.get_team_stats(team_id, league, force_refresh=True)
                    teams_processed.add(team_id)

        logger.info(f"✅ Cache warmed for {len(teams_processed)} teams")


# Global data manager instance (singleton)
_data_manager_instance: Optional[DataManager] = None


def get_data_manager() -> DataManager:
    """Get global data manager instance"""
    global _data_manager_instance

    if _data_manager_instance is None:
        _data_manager_instance = DataManager()

    return _data_manager_instance


if __name__ == "__main__":
    # Test data manager
    print("Testing Data Manager...")

    dm = DataManager()

    # Test team stats
    print("\n1. Testing team stats retrieval...")
    stats = dm.get_team_stats('lakers', 'NBA')
    if stats:
        print(f"✅ Retrieved stats for Lakers:")
        print(f"   Record: {stats['wins']}-{stats['losses']}")
        print(f"   PPG: {stats['ppg']:.1f}")
        print(f"   Elo: {stats['elo_rating']:.0f}")
    else:
        print("⚠️  No stats found (expected if database is empty)")

    # Test today's games
    print("\n2. Testing today's games...")
    games = dm.get_todays_games('NBA')
    print(f"   Found {len(games)} games today")

    print("\n✅ Data Manager test complete!")
