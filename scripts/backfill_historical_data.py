#!/usr/bin/env python3
"""
Historical Data Backfill Script for GameLens.ai
Fetches 3+ years of historical game data from Goalserve API

Usage:
    python scripts/backfill_historical_data.py --league NBA --start-year 2010 --end-year 2024
    python scripts/backfill_historical_data.py --league NFL --start-year 2015 --end-year 2024 --resume
    python scripts/backfill_historical_data.py --all-leagues --start-year 2020
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
import time
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from src.data.goalserve_client import GoalserveClient
from src.models.database_schema import Base, GamesHistory, OddsHistory, create_all_tables
from config.redis_config import get_cache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataBackfill:
    """
    Backfill historical game data from Goalserve API

    Strategy:
    - Fetch games month by month to avoid rate limits
    - Resume capability (tracks progress)
    - Validates and deduplicate data
    - Stores games + odds separately
    """

    SUPPORTED_LEAGUES = ['NBA', 'NFL', 'MLB', 'NHL', 'NCAAF', 'NCAAB']

    # Season start months by league
    SEASON_START_MONTHS = {
        'NBA': 10,      # October
        'NFL': 9,       # September
        'MLB': 3,       # March/April
        'NHL': 10,      # October
        'NCAAF': 8,     # August
        'NCAAB': 11     # November
    }

    def __init__(self, db_url: str, goalserve_api_key: str):
        """
        Initialize backfill manager

        Args:
            db_url: PostgreSQL database URL
            goalserve_api_key: Goalserve API key
        """
        self.engine = create_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)
        self.goalserve = GoalserveClient(api_key=goalserve_api_key)

        self.progress_file = Path("backfill_progress.json")
        self.progress = self._load_progress()

        # Stats
        self.stats = {
            'games_added': 0,
            'games_updated': 0,
            'games_skipped': 0,
            'odds_added': 0,
            'errors': 0
        }

    def _load_progress(self) -> Dict:
        """Load progress from file to enable resume"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_progress(self, league: str, year: int, month: int):
        """Save current progress"""
        key = f"{league}_{year}"
        if key not in self.progress:
            self.progress[key] = {'completed_months': []}

        self.progress[key]['completed_months'].append(month)

        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def _is_month_completed(self, league: str, year: int, month: int) -> bool:
        """Check if month was already processed"""
        key = f"{league}_{year}"
        if key in self.progress:
            return month in self.progress[key].get('completed_months', [])
        return False

    def backfill_league(self, league: str, start_year: int, end_year: int, resume: bool = True):
        """
        Backfill historical data for a league

        Args:
            league: League code (NBA, NFL, etc.)
            start_year: Starting year (e.g., 2010)
            end_year: Ending year (e.g., 2024)
            resume: If True, skip already processed months
        """
        logger.info(f"=" * 80)
        logger.info(f"BACKFILLING {league}: {start_year} - {end_year}")
        logger.info(f"=" * 80)

        # Get season structure
        season_start_month = self.SEASON_START_MONTHS.get(league, 1)

        for year in range(start_year, end_year + 1):
            logger.info(f"\n📅 Processing {league} {year} season...")

            # Fetch games for entire year
            for month in range(1, 13):
                # Skip if already completed and resume enabled
                if resume and self._is_month_completed(league, year, month):
                    logger.info(f"  ⏭️  Skipping {year}-{month:02d} (already completed)")
                    continue

                try:
                    self._fetch_month_games(league, year, month)
                    self._save_progress(league, year, month)

                    # Rate limiting (1 second between requests)
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"  ❌ Error processing {year}-{month:02d}: {e}")
                    self.stats['errors'] += 1
                    continue

            logger.info(f"✅ Completed {league} {year}")

        # Print final stats
        self._print_stats()

    def _fetch_month_games(self, league: str, year: int, month: int):
        """Fetch and store games for a specific month"""
        logger.info(f"  📥 Fetching {league} {year}-{month:02d}...")

        # Get start and end dates for the month
        start_date = date(year, month, 1)

        # Last day of month
        if month == 12:
            end_date = date(year, 12, 31)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)

        # Fetch games for each day in the month
        current_date = start_date
        games_this_month = 0

        while current_date <= end_date:
            try:
                # Fetch games from Goalserve
                games = self.goalserve.get_scores(
                    sport=league.lower(),
                    date=current_date.strftime('%d.%m.%Y')
                )

                # Store each game
                for game in games:
                    if self._store_game(game, league):
                        games_this_month += 1

                # Small delay between requests
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"    ⚠️  Error fetching {current_date}: {e}")

            current_date += timedelta(days=1)

        logger.info(f"    ✅ Stored {games_this_month} games")

    def _store_game(self, game_data: Dict, league: str) -> bool:
        """
        Store game in database

        Args:
            game_data: Game data from Goalserve
            league: League code

        Returns:
            True if stored successfully
        """
        session = self.Session()

        try:
            # Check if game already exists
            existing = session.query(GamesHistory).filter_by(
                game_id=game_data['game_id']
            ).first()

            if existing:
                # Update if new information available
                if game_data.get('status') == 'final' and existing.status != 'final':
                    self._update_game(session, existing, game_data)
                    self.stats['games_updated'] += 1
                else:
                    self.stats['games_skipped'] += 1
                return False

            # Create new game record
            game = GamesHistory(
                game_id=game_data['game_id'],
                league=league,
                season=self._determine_season(game_data['game_time'], league),
                game_date=game_data['game_time'].date(),
                game_datetime=game_data['game_time'],

                home_team_id=game_data['home_team'],
                away_team_id=game_data['away_team'],
                home_team_name=game_data['home_team'],
                away_team_name=game_data['away_team'],

                home_score=game_data.get('home_score'),
                away_score=game_data.get('away_score'),

                status=game_data.get('status', 'scheduled'),
                venue=game_data.get('venue'),

                spread=game_data.get('spread'),
                total=game_data.get('total'),
                home_moneyline=game_data.get('home_odds'),
                away_moneyline=game_data.get('away_odds'),

                data_source='goalserve'
            )

            # Calculate betting outcomes if game is final
            if game.status == 'final' and game.home_score is not None:
                game.home_covered = self._calculate_cover(
                    game.home_score, game.away_score, game.spread
                )
                game.went_over = self._calculate_over(
                    game.home_score, game.away_score, game.total
                )

            session.add(game)
            session.commit()

            self.stats['games_added'] += 1
            return True

        except IntegrityError:
            session.rollback()
            self.stats['games_skipped'] += 1
            return False

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing game: {e}")
            self.stats['errors'] += 1
            return False

        finally:
            session.close()

    def _update_game(self, session, game: GamesHistory, game_data: Dict):
        """Update existing game with new information"""
        game.home_score = game_data.get('home_score', game.home_score)
        game.away_score = game_data.get('away_score', game.away_score)
        game.status = game_data.get('status', game.status)

        if game.status == 'final' and game.home_score is not None:
            game.home_covered = self._calculate_cover(
                game.home_score, game.away_score, game.spread
            )
            game.went_over = self._calculate_over(
                game.home_score, game.away_score, game.total
            )

        game.updated_at = datetime.utcnow()
        session.commit()

    def _determine_season(self, game_date: datetime, league: str) -> int:
        """Determine season year based on game date"""
        season_start_month = self.SEASON_START_MONTHS.get(league, 1)

        if game_date.month >= season_start_month:
            return game_date.year
        else:
            return game_date.year - 1

    def _calculate_cover(self, home_score: int, away_score: int, spread: Optional[float]) -> Optional[bool]:
        """Calculate if home team covered the spread"""
        if spread is None:
            return None

        # Spread is negative if home favored
        # e.g., spread = -5.5 means home needs to win by 6+
        home_margin = home_score - away_score
        return home_margin + spread > 0

    def _calculate_over(self, home_score: int, away_score: int, total: Optional[float]) -> Optional[bool]:
        """Calculate if game went over the total"""
        if total is None:
            return None

        return (home_score + away_score) > total

    def _print_stats(self):
        """Print backfill statistics"""
        logger.info(f"\n" + "=" * 80)
        logger.info(f"BACKFILL COMPLETE")
        logger.info(f"=" * 80)
        logger.info(f"Games added:   {self.stats['games_added']:,}")
        logger.info(f"Games updated: {self.stats['games_updated']:,}")
        logger.info(f"Games skipped: {self.stats['games_skipped']:,}")
        logger.info(f"Odds records:  {self.stats['odds_added']:,}")
        logger.info(f"Errors:        {self.stats['errors']:,}")
        logger.info(f"=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Backfill historical game data from Goalserve API'
    )

    parser.add_argument(
        '--league',
        type=str,
        choices=['NBA', 'NFL', 'MLB', 'NHL', 'NCAAF', 'NCAAB', 'all'],
        default='NBA',
        help='League to backfill (or "all" for all leagues)'
    )

    parser.add_argument(
        '--start-year',
        type=int,
        default=2020,
        help='Starting year (default: 2020)'
    )

    parser.add_argument(
        '--end-year',
        type=int,
        default=datetime.now().year,
        help='Ending year (default: current year)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint (skip already processed months)'
    )

    parser.add_argument(
        '--db-url',
        type=str,
        default=os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/gamelens'),
        help='Database URL (default from DATABASE_URL env)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('GOALSERVE_API_KEY'),
        help='Goalserve API key (default from GOALSERVE_API_KEY env)'
    )

    parser.add_argument(
        '--create-tables',
        action='store_true',
        help='Create database tables before backfilling'
    )

    args = parser.parse_args()

    # Validate API key
    if not args.api_key:
        logger.error("❌ No Goalserve API key provided. Set GOALSERVE_API_KEY env var or use --api-key")
        sys.exit(1)

    # Create tables if requested
    if args.create_tables:
        logger.info("Creating database tables...")
        engine = create_engine(args.db_url)
        create_all_tables(engine)
        logger.info("✅ Tables created")

    # Initialize backfill manager
    backfill = HistoricalDataBackfill(
        db_url=args.db_url,
        goalserve_api_key=args.api_key
    )

    # Determine leagues to process
    leagues = (
        HistoricalDataBackfill.SUPPORTED_LEAGUES
        if args.league == 'all'
        else [args.league]
    )

    # Process each league
    for league in leagues:
        backfill.backfill_league(
            league=league,
            start_year=args.start_year,
            end_year=args.end_year,
            resume=args.resume
        )

    logger.info("\n✅ All leagues processed successfully!")


if __name__ == "__main__":
    main()
