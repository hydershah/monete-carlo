"""
Historical Data Fetcher for Backtesting
=========================================
Fetches historical game data from ESPN API to populate the backtesting system.
Supports fetching multiple seasons of data for NBA, NFL, MLB, NHL.

Usage:
    from src.data.historical_data_fetcher import HistoricalDataFetcher

    fetcher = HistoricalDataFetcher()

    # Fetch last 3 NBA seasons
    games = fetcher.fetch_historical_games('nba', seasons=3)

    # Save to database
    fetcher.save_to_database(games)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
import time
from tqdm import tqdm
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .espn_client import ESPNClient
from ..models.database_schema import GamesHistory, TeamStatsDaily, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

logger = logging.getLogger(__name__)


@dataclass
class HistoricalGame:
    """Standardized historical game format for backtesting"""
    game_id: str
    game_date: datetime
    league: str
    season: int

    # Teams
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str

    # Scores
    home_score: int
    away_score: int

    # Game info
    status: str
    venue: Optional[str] = None
    neutral_site: bool = False

    # Betting lines (if available)
    spread: Optional[float] = None
    total: Optional[float] = None
    home_ml: Optional[float] = None
    away_ml: Optional[float] = None

    # Additional metadata
    completed: bool = True
    playoff: bool = False


class HistoricalDataFetcher:
    """
    Fetches historical sports data from ESPN API for backtesting.
    Handles multi-season data collection with proper rate limiting.
    """

    # Season date ranges for each sport
    SEASON_CONFIGS = {
        'nba': {
            'start_month': 10,  # October
            'end_month': 6,      # June (next year)
            'typical_games': 1230,  # ~82 games per team * 30 teams / 2
        },
        'nfl': {
            'start_month': 9,   # September
            'end_month': 2,     # February (next year)
            'typical_games': 285,  # 272 regular + playoffs
        },
        'mlb': {
            'start_month': 3,   # March/April
            'end_month': 10,    # October
            'typical_games': 2430,  # 162 games per team * 30 teams / 2
        },
        'nhl': {
            'start_month': 10,  # October
            'end_month': 6,     # June (next year)
            'typical_games': 1312,  # 82 games per team * 32 teams / 2
        }
    }

    def __init__(self, db_url: Optional[str] = None, rate_limit_delay: float = 1.0):
        """
        Initialize historical data fetcher.

        Args:
            db_url: Database connection string
            rate_limit_delay: Delay between API requests (seconds)
        """
        self.espn_client = ESPNClient(rate_limit_delay=rate_limit_delay)

        # Database setup
        self.db_url = db_url or os.getenv(
            'DATABASE_URL',
            'postgresql://user:password@localhost:5432/gamelens'
        )

        try:
            self.engine = create_engine(self.db_url, echo=False, pool_pre_ping=True)
            self.Session = sessionmaker(bind=self.engine)
            logger.info(f"✅ Connected to database for historical data ingestion")
        except Exception as e:
            logger.warning(f"⚠️  Database connection failed: {e}")
            logger.info("Will fetch data but cannot save to database")
            self.engine = None
            self.Session = None

    def fetch_historical_games(
        self,
        league: str,
        seasons: int = 3,
        start_year: Optional[int] = None
    ) -> List[HistoricalGame]:
        """
        Fetch historical games for multiple seasons.

        Args:
            league: League identifier (nba, nfl, mlb, nhl)
            seasons: Number of seasons to fetch (default: 3)
            start_year: Starting year (default: current year - seasons)

        Returns:
            List of HistoricalGame objects
        """
        if league.lower() not in self.SEASON_CONFIGS:
            raise ValueError(f"Unsupported league: {league}")

        config = self.SEASON_CONFIGS[league.lower()]

        # Determine season years
        current_year = datetime.now().year
        if not start_year:
            start_year = current_year - seasons

        all_games = []

        logger.info(f"🏀 Fetching {seasons} seasons of {league.upper()} data")
        logger.info(f"📅 Years: {start_year} to {current_year}")

        for year in range(start_year, current_year + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Fetching {league.upper()} {year}-{year+1} season")
            logger.info(f"{'='*60}")

            season_games = self._fetch_season_games(league, year, config)

            if season_games:
                all_games.extend(season_games)
                logger.info(f"✅ Fetched {len(season_games)} games for {year}-{year+1}")
            else:
                logger.warning(f"⚠️  No games found for {year}-{year+1}")

        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 TOTAL: Fetched {len(all_games)} games across {seasons} seasons")
        logger.info(f"{'='*60}\n")

        return all_games

    def _fetch_season_games(
        self,
        league: str,
        start_year: int,
        config: Dict
    ) -> List[HistoricalGame]:
        """
        Fetch all games for a single season.

        Args:
            league: League identifier
            start_year: Season start year
            config: Season configuration

        Returns:
            List of games for the season
        """
        season_games = []

        # Calculate season date range
        season_start = datetime(start_year, config['start_month'], 1)

        # Handle seasons that span years
        end_year = start_year if config['end_month'] > config['start_month'] else start_year + 1
        season_end = datetime(end_year, config['end_month'], 28)  # Safe end day

        # Add extra days for playoffs
        if league.lower() in ['nba', 'nhl']:
            season_end += timedelta(days=30)
        elif league.lower() == 'nfl':
            season_end += timedelta(days=45)  # Include Super Bowl

        logger.info(f"   Date range: {season_start.date()} to {season_end.date()}")

        # Fetch games day by day (ESPN API is date-based)
        current_date = season_start
        total_days = (season_end - season_start).days

        # Use tqdm for progress bar
        with tqdm(total=total_days, desc=f"   Fetching {league.upper()} games", unit="day") as pbar:
            while current_date <= season_end:
                date_str = current_date.strftime("%Y%m%d")

                try:
                    # Fetch games for this date
                    scoreboard = self.espn_client.get_scoreboard(league, date=date_str)
                    daily_games = self.espn_client.parse_games(scoreboard)

                    # Convert to HistoricalGame format
                    for game in daily_games:
                        if game['completed']:  # Only completed games for backtesting
                            hist_game = self._convert_to_historical_game(
                                game,
                                league,
                                start_year
                            )
                            if hist_game:
                                season_games.append(hist_game)

                except Exception as e:
                    logger.debug(f"   No games on {date_str}: {e}")

                current_date += timedelta(days=1)
                pbar.update(1)

        return season_games

    def _convert_to_historical_game(
        self,
        espn_game: Dict[str, Any],
        league: str,
        season: int
    ) -> Optional[HistoricalGame]:
        """
        Convert ESPN game format to HistoricalGame format.

        Args:
            espn_game: Game data from ESPN API
            league: League identifier
            season: Season year

        Returns:
            HistoricalGame object or None if invalid
        """
        try:
            # Parse game date
            game_date = datetime.fromisoformat(espn_game['date'].replace('Z', '+00:00'))

            # Extract odds if available
            spread = None
            total = None
            home_ml = None
            away_ml = None

            if espn_game.get('odds'):
                odds = espn_game['odds'][0] if isinstance(espn_game['odds'], list) else espn_game['odds']
                spread = odds.get('spread')
                total = odds.get('overUnder')
                home_ml = odds.get('homeTeamOdds', {}).get('moneyLine')
                away_ml = odds.get('awayTeamOdds', {}).get('moneyLine')

            return HistoricalGame(
                game_id=espn_game['game_id'],
                game_date=game_date,
                league=league.upper(),
                season=season,

                home_team_id=espn_game['home_team_id'],
                home_team_name=espn_game['home_team_name'],
                away_team_id=espn_game['away_team_id'],
                away_team_name=espn_game['away_team_name'],

                home_score=espn_game['home_score'],
                away_score=espn_game['away_score'],

                status=espn_game['status'],
                venue=espn_game.get('venue'),
                neutral_site=False,  # Could parse from venue info

                spread=spread,
                total=total,
                home_ml=home_ml,
                away_ml=away_ml,

                completed=espn_game['completed'],
                playoff=False  # Could parse from game name
            )

        except Exception as e:
            logger.error(f"Error converting game {espn_game.get('game_id')}: {e}")
            return None

    def save_to_database(self, games: List[HistoricalGame]) -> int:
        """
        Save historical games to database.

        Args:
            games: List of HistoricalGame objects

        Returns:
            Number of games saved
        """
        if not self.Session:
            logger.error("❌ No database connection - cannot save games")
            return 0

        session = self.Session()
        saved_count = 0

        try:
            logger.info(f"💾 Saving {len(games)} games to database...")

            for game in tqdm(games, desc="Saving to DB", unit="game"):
                try:
                    # Check if game already exists
                    existing = session.query(GamesHistory).filter_by(
                        game_id=game.game_id
                    ).first()

                    if existing:
                        # Update existing game
                        for key, value in asdict(game).items():
                            if hasattr(existing, key):
                                setattr(existing, key, value)
                    else:
                        # Create new game record
                        db_game = GamesHistory(
                            game_id=game.game_id,
                            game_date=game.game_date.date(),
                            league=game.league,
                            season=game.season,

                            home_team_id=game.home_team_id,
                            home_team_name=game.home_team_name,
                            away_team_id=game.away_team_id,
                            away_team_name=game.away_team_name,

                            home_score=game.home_score,
                            away_score=game.away_score,

                            status='final' if game.completed else game.status,
                            venue=game.venue,
                            neutral_site=game.neutral_site,

                            spread=game.spread,
                            total=game.total,

                            # Calculate derived fields
                            home_win=game.home_score > game.away_score,
                            home_covered=(game.home_score + (game.spread or 0)) > game.away_score if game.spread else None,
                            went_over=(game.home_score + game.away_score) > game.total if game.total else None
                        )
                        session.add(db_game)

                    saved_count += 1

                    # Commit in batches
                    if saved_count % 100 == 0:
                        session.commit()

                except Exception as e:
                    logger.error(f"Error saving game {game.game_id}: {e}")
                    session.rollback()

            # Final commit
            session.commit()
            logger.info(f"✅ Successfully saved {saved_count} games to database")

        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            session.rollback()

        finally:
            session.close()

        return saved_count

    def save_to_csv(self, games: List[HistoricalGame], filepath: str):
        """
        Save historical games to CSV file.

        Args:
            games: List of HistoricalGame objects
            filepath: Output CSV file path
        """
        df = pd.DataFrame([asdict(g) for g in games])
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Saved {len(games)} games to {filepath}")

    def load_from_csv(self, filepath: str) -> List[HistoricalGame]:
        """
        Load historical games from CSV file.

        Args:
            filepath: Input CSV file path

        Returns:
            List of HistoricalGame objects
        """
        df = pd.read_csv(filepath)
        df['game_date'] = pd.to_datetime(df['game_date'])

        games = []
        for _, row in df.iterrows():
            games.append(HistoricalGame(**row.to_dict()))

        logger.info(f"📂 Loaded {len(games)} games from {filepath}")
        return games

    def get_games_dataframe(self, games: List[HistoricalGame]) -> pd.DataFrame:
        """
        Convert games to pandas DataFrame for backtesting.

        Args:
            games: List of HistoricalGame objects

        Returns:
            DataFrame formatted for backtesting system
        """
        df = pd.DataFrame([asdict(g) for g in games])

        # Add derived columns expected by backtesting
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        df['point_diff'] = df['home_score'] - df['away_score']

        # Sort by date
        df = df.sort_values('game_date').reset_index(drop=True)

        return df

    def validate_data_quality(self, games: List[HistoricalGame]) -> Dict[str, Any]:
        """
        Validate the quality of fetched historical data.

        Args:
            games: List of HistoricalGame objects

        Returns:
            Dictionary with validation statistics
        """
        df = pd.DataFrame([asdict(g) for g in games])

        validation = {
            'total_games': len(games),
            'date_range': {
                'start': df['game_date'].min(),
                'end': df['game_date'].max(),
                'span_days': (df['game_date'].max() - df['game_date'].min()).days
            },
            'seasons_covered': df['season'].nunique(),
            'leagues': df['league'].unique().tolist(),
            'teams_count': len(set(df['home_team_id'].tolist() + df['away_team_id'].tolist())),
            'completeness': {
                'missing_scores': df[['home_score', 'away_score']].isna().sum().sum(),
                'missing_odds': df[['spread', 'total']].isna().sum().to_dict(),
                'games_per_season': df.groupby('season').size().to_dict()
            },
            'quality_checks': {
                'duplicate_games': df.duplicated(subset=['game_id']).sum(),
                'invalid_scores': len(df[(df['home_score'] < 0) | (df['away_score'] < 0)]),
                'future_dates': len(df[df['game_date'] > pd.Timestamp.now(tz='UTC')])
            }
        }

        return validation


def main():
    """
    Main function to fetch and save historical data.
    Run this script to populate your database with backtesting data.
    """
    print("\n" + "="*60)
    print("🏀 HISTORICAL DATA FETCHER FOR BACKTESTING")
    print("="*60 + "\n")

    # Initialize fetcher
    fetcher = HistoricalDataFetcher(rate_limit_delay=1.0)

    # Fetch NBA data (last 3 seasons)
    league = 'nba'
    seasons = 3

    print(f"Fetching {seasons} seasons of {league.upper()} data...")
    print(f"This may take 10-15 minutes due to rate limiting...\n")

    games = fetcher.fetch_historical_games(league, seasons=seasons)

    # Validate data
    print("\n" + "="*60)
    print("📊 DATA VALIDATION")
    print("="*60 + "\n")

    validation = fetcher.validate_data_quality(games)
    print(f"Total games: {validation['total_games']}")
    print(f"Date range: {validation['date_range']['start']} to {validation['date_range']['end']}")
    print(f"Seasons: {validation['seasons_covered']}")
    print(f"Teams: {validation['teams_count']}")
    print(f"Duplicate games: {validation['quality_checks']['duplicate_games']}")
    print(f"Games per season: {validation['completeness']['games_per_season']}")

    # Save to CSV as backup
    csv_path = f"data/{league}_historical_{seasons}seasons.csv"
    os.makedirs("data", exist_ok=True)
    fetcher.save_to_csv(games, csv_path)

    # Save to database
    print(f"\n{'='*60}")
    print("💾 SAVING TO DATABASE")
    print("="*60 + "\n")

    saved_count = fetcher.save_to_database(games)

    print(f"\n{'='*60}")
    print("✅ COMPLETE")
    print("="*60)
    print(f"✅ Fetched: {len(games)} games")
    print(f"✅ Saved to DB: {saved_count} games")
    print(f"✅ Backup CSV: {csv_path}")
    print("\n🎉 Historical data is ready for backtesting!\n")


if __name__ == "__main__":
    main()
