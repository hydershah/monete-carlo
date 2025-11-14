#!/usr/bin/env python3
"""
Daily Update Jobs for GameLens.ai Production
Keeps database and cache current with latest games and stats

Jobs:
1. Update yesterday's completed games (3 AM daily)
2. Refresh team stats cache (every 6 hours)
3. Update live odds (every 15 minutes during game hours)

Usage:
    # Run all jobs
    python scripts/daily_update_job.py --job all

    # Run specific job
    python scripts/daily_update_job.py --job update_games
    python scripts/daily_update_job.py --job refresh_cache
    python scripts/daily_update_job.py --job update_odds

    # Schedule with cron:
    0 3 * * * cd /path/to/project && python scripts/daily_update_job.py --job update_games
    0 */6 * * * cd /path/to/project && python scripts/daily_update_job.py --job refresh_cache
    */15 * * * * cd /path/to/project && python scripts/daily_update_job.py --job update_odds
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from src.data.goalserve_client import GoalserveClient
from src.data.espn_client import ESPNClient
from src.models.database_schema import GamesHistory, TeamStatsDaily, TeamRatingsHistory, OddsHistory
from src.simulations.enhanced_elo_model import EnhancedEloModel
from src.simulations.pythagorean_expectations import PythagoreanExpectations, TeamStats
from config.redis_config import get_cache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DailyUpdateJobs:
    """Daily update jobs for production data pipeline"""

    SUPPORTED_LEAGUES = ['NBA', 'NFL', 'MLB', 'NHL']

    def __init__(self, db_url: str):
        """
        Initialize update job manager

        Args:
            db_url: PostgreSQL database URL
        """
        self.engine = create_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)

        self.espn = ESPNClient()
        self.goalserve = GoalserveClient(api_key=os.getenv('GOALSERVE_API_KEY'))
        self.cache = get_cache()

        self.stats = {
            'games_updated': 0,
            'teams_updated': 0,
            'ratings_updated': 0,
            'odds_updated': 0,
            'errors': 0
        }

    # ==================== JOB 1: UPDATE GAMES ====================

    def update_yesterday_games(self):
        """
        Fetch and store yesterday's completed games
        Run daily at 3 AM (after all games complete)
        """
        logger.info("=" * 80)
        logger.info("JOB 1: UPDATING YESTERDAY'S GAMES")
        logger.info("=" * 80)

        yesterday = date.today() - timedelta(days=1)
        logger.info(f"Processing date: {yesterday}")

        for league in self.SUPPORTED_LEAGUES:
            try:
                logger.info(f"\n📥 Fetching {league} games for {yesterday}...")

                # Fetch from ESPN (primary source for recent games)
                games = self.espn.get_scoreboard(
                    league=league.lower(),
                    date=yesterday.strftime('%Y%m%d')
                )

                # Store each game
                for game in games:
                    self._store_or_update_game(game, league)

                logger.info(f"  ✅ Processed {len(games)} {league} games")

            except Exception as e:
                logger.error(f"  ❌ Error processing {league}: {e}")
                self.stats['errors'] += 1

        # Update Elo ratings after all games processed
        self._update_elo_ratings(yesterday)

        # Print stats
        logger.info(f"\n✅ Games update complete:")
        logger.info(f"   Games updated: {self.stats['games_updated']}")
        logger.info(f"   Ratings updated: {self.stats['ratings_updated']}")

    def _store_or_update_game(self, game_data: Dict, league: str):
        """Store or update game in database"""
        session = self.Session()

        try:
            existing = session.query(GamesHistory).filter_by(
                game_id=game_data['id']
            ).first()

            if existing:
                # Update existing game
                existing.home_score = game_data.get('home_score', existing.home_score)
                existing.away_score = game_data.get('away_score', existing.away_score)
                existing.status = game_data.get('status', existing.status)

                # Calculate betting outcomes
                if existing.status == 'final' and existing.home_score is not None:
                    existing.home_covered = self._calculate_cover(
                        existing.home_score, existing.away_score, existing.spread
                    )
                    existing.went_over = self._calculate_over(
                        existing.home_score, existing.away_score, existing.total
                    )

                existing.updated_at = datetime.utcnow()
                self.stats['games_updated'] += 1

            else:
                # Create new game
                game = GamesHistory(
                    game_id=game_data['id'],
                    league=league,
                    season=game_data.get('season', datetime.now().year),
                    game_date=game_data['date'].date() if isinstance(game_data['date'], datetime) else game_data['date'],
                    game_datetime=game_data['date'],

                    home_team_id=game_data['home_team_id'],
                    away_team_id=game_data['away_team_id'],
                    home_team_name=game_data['home_team'],
                    away_team_name=game_data['away_team'],

                    home_score=game_data.get('home_score'),
                    away_score=game_data.get('away_score'),
                    status=game_data.get('status', 'scheduled'),

                    spread=game_data.get('spread'),
                    total=game_data.get('total'),

                    data_source='espn'
                )

                session.add(game)
                self.stats['games_updated'] += 1

            session.commit()

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing game: {e}")
            self.stats['errors'] += 1

        finally:
            session.close()

    def _update_elo_ratings(self, date_to_update: date):
        """Update Elo ratings based on yesterday's games"""
        session = self.Session()

        try:
            for league in self.SUPPORTED_LEAGUES:
                logger.info(f"\n📊 Updating {league} Elo ratings...")

                # Get completed games for the date
                games = session.query(GamesHistory).filter(
                    GamesHistory.league == league,
                    GamesHistory.game_date == date_to_update,
                    GamesHistory.status == 'final'
                ).all()

                if not games:
                    continue

                # Initialize Elo model
                elo_model = EnhancedEloModel(sport=league)

                for game in games:
                    # Get current ratings (or start at 1500)
                    home_rating = self._get_latest_elo(session, game.home_team_id, league) or 1500
                    away_rating = self._get_latest_elo(session, game.away_team_id, league) or 1500

                    # Update ratings based on game result
                    margin = game.home_score - game.away_score
                    new_home, new_away = elo_model.update_ratings(
                        home_rating=home_rating,
                        away_rating=away_rating,
                        home_score=game.home_score,
                        away_score=game.away_score,
                        home_won=(margin > 0)
                    )

                    # Store new ratings
                    self._store_elo_rating(session, game.home_team_id, league, new_home, date_to_update)
                    self._store_elo_rating(session, game.away_team_id, league, new_away, date_to_update)

                    # Cache the ratings
                    self.cache.cache_elo_rating(game.home_team_id, league, new_home)
                    self.cache.cache_elo_rating(game.away_team_id, league, new_away)

                    self.stats['ratings_updated'] += 2

                logger.info(f"  ✅ Updated {self.stats['ratings_updated']} ratings")

            session.commit()

        except Exception as e:
            session.rollback()
            logger.error(f"Error updating Elo ratings: {e}")
            self.stats['errors'] += 1

        finally:
            session.close()

    def _get_latest_elo(self, session, team_id: str, league: str) -> Optional[float]:
        """Get team's most recent Elo rating"""
        rating = session.query(TeamRatingsHistory).filter(
            TeamRatingsHistory.team_id == team_id,
            TeamRatingsHistory.league == league
        ).order_by(TeamRatingsHistory.rating_date.desc()).first()

        return rating.elo_rating if rating else None

    def _store_elo_rating(self, session, team_id: str, league: str, rating: float, rating_date: date):
        """Store new Elo rating"""
        rating_record = TeamRatingsHistory(
            team_id=team_id,
            league=league,
            rating_date=rating_date,
            season=datetime.now().year,
            elo_rating=rating
        )
        session.add(rating_record)

    # ==================== JOB 2: REFRESH CACHE ====================

    def refresh_team_stats_cache(self):
        """
        Refresh team stats in Redis cache
        Run every 6 hours to keep current season stats fresh
        """
        logger.info("=" * 80)
        logger.info("JOB 2: REFRESHING TEAM STATS CACHE")
        logger.info("=" * 80)

        session = self.Session()

        try:
            for league in self.SUPPORTED_LEAGUES:
                logger.info(f"\n🔄 Refreshing {league} team stats...")

                # Get all teams in the league
                teams = session.query(GamesHistory.home_team_id).filter(
                    GamesHistory.league == league,
                    GamesHistory.season == datetime.now().year
                ).distinct().all()

                for (team_id,) in teams:
                    # Calculate current season stats
                    stats = self._calculate_team_stats(session, team_id, league)

                    if stats:
                        # Cache in Redis
                        self.cache.cache_team_stats(team_id, league, stats)

                        # Store daily snapshot in DB
                        self._store_team_stats_snapshot(session, team_id, league, stats)

                        self.stats['teams_updated'] += 1

                logger.info(f"  ✅ Cached stats for {self.stats['teams_updated']} teams")

            session.commit()

        except Exception as e:
            session.rollback()
            logger.error(f"Error refreshing cache: {e}")
            self.stats['errors'] += 1

        finally:
            session.close()

        logger.info(f"\n✅ Cache refresh complete:")
        logger.info(f"   Teams updated: {self.stats['teams_updated']}")

    def _calculate_team_stats(self, session, team_id: str, league: str) -> Optional[Dict]:
        """Calculate current season stats for a team"""
        try:
            # Get all games for current season
            current_season = datetime.now().year

            home_games = session.query(GamesHistory).filter(
                GamesHistory.home_team_id == team_id,
                GamesHistory.league == league,
                GamesHistory.season == current_season,
                GamesHistory.status == 'final'
            ).all()

            away_games = session.query(GamesHistory).filter(
                GamesHistory.away_team_id == team_id,
                GamesHistory.league == league,
                GamesHistory.season == current_season,
                GamesHistory.status == 'final'
            ).all()

            if not (home_games or away_games):
                return None

            # Calculate stats
            wins = 0
            losses = 0
            points_for = 0
            points_against = 0

            for game in home_games:
                points_for += game.home_score
                points_against += game.away_score
                if game.home_score > game.away_score:
                    wins += 1
                else:
                    losses += 1

            for game in away_games:
                points_for += game.away_score
                points_against += game.home_score
                if game.away_score > game.home_score:
                    wins += 1
                else:
                    losses += 1

            games_played = len(home_games) + len(away_games)

            # Get latest Elo rating
            elo_rating = self._get_latest_elo(session, team_id, league) or 1500

            # Calculate Pythagorean expectation
            pythag_model = PythagoreanExpectations(sport=league)
            pythag_win_pct = pythag_model.calculate_expected_win_pct(points_for, points_against)

            return {
                'team_id': team_id,
                'league': league,
                'wins': wins,
                'losses': losses,
                'games_played': games_played,
                'win_pct': wins / games_played if games_played > 0 else 0,
                'points_for': points_for,
                'points_against': points_against,
                'ppg': points_for / games_played if games_played > 0 else 0,
                'papg': points_against / games_played if games_played > 0 else 0,
                'elo_rating': elo_rating,
                'pythagorean_win_pct': pythag_win_pct,
                'updated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating stats for {team_id}: {e}")
            return None

    def _store_team_stats_snapshot(self, session, team_id: str, league: str, stats: Dict):
        """Store daily snapshot of team stats"""
        snapshot = TeamStatsDaily(
            team_id=team_id,
            league=league,
            snapshot_date=date.today(),
            season=datetime.now().year,

            wins=stats['wins'],
            losses=stats['losses'],
            games_played=stats['games_played'],
            win_pct=stats['win_pct'],

            points_for=stats['points_for'],
            points_against=stats['points_against'],
            ppg=stats['ppg'],
            papg=stats['papg'],

            elo_rating=stats['elo_rating'],
            pythagorean_win_pct=stats['pythagorean_win_pct']
        )

        session.merge(snapshot)  # Upsert

    # ==================== JOB 3: UPDATE ODDS ====================

    def update_live_odds(self):
        """
        Update current odds for upcoming games
        Run every 15 minutes during betting hours
        """
        logger.info("=" * 80)
        logger.info("JOB 3: UPDATING LIVE ODDS")
        logger.info("=" * 80)

        # This would integrate with TheOdds API or Goalserve odds endpoint
        # For now, placeholder implementation

        logger.info("⚠️  Odds update not fully implemented - requires API integration")
        logger.info("    To implement: Use TheOddsAPI.get_odds() or Goalserve odds endpoints")

    # ==================== UTILITY METHODS ====================

    def _calculate_cover(self, home_score: int, away_score: int, spread: Optional[float]) -> Optional[bool]:
        """Calculate if home team covered spread"""
        if spread is None:
            return None
        home_margin = home_score - away_score
        return home_margin + spread > 0

    def _calculate_over(self, home_score: int, away_score: int, total: Optional[float]) -> Optional[bool]:
        """Calculate if game went over total"""
        if total is None:
            return None
        return (home_score + away_score) > total


def main():
    parser = argparse.ArgumentParser(description='Daily update jobs for GameLens.ai')

    parser.add_argument(
        '--job',
        type=str,
        choices=['update_games', 'refresh_cache', 'update_odds', 'all'],
        required=True,
        help='Job to run'
    )

    parser.add_argument(
        '--db-url',
        type=str,
        default=os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/gamelens'),
        help='Database URL'
    )

    args = parser.parse_args()

    # Initialize job manager
    jobs = DailyUpdateJobs(db_url=args.db_url)

    # Run requested job
    if args.job == 'update_games' or args.job == 'all':
        jobs.update_yesterday_games()

    if args.job == 'refresh_cache' or args.job == 'all':
        jobs.refresh_team_stats_cache()

    if args.job == 'update_odds' or args.job == 'all':
        jobs.update_live_odds()

    logger.info("\n✅ All jobs completed successfully!")


if __name__ == "__main__":
    main()
