"""
Data ingestion module for fetching and storing sports data.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import and_

from .espn_client import ESPNClient
from .theodds_client import TheOddsAPIClient
from ..models import Team, Game, get_db_context


class DataIngestion:
    """Handles fetching and storing sports data from various APIs."""

    def __init__(
        self,
        espn_client: Optional[ESPNClient] = None,
        odds_client: Optional[TheOddsAPIClient] = None
    ):
        """
        Initialize data ingestion.

        Args:
            espn_client: ESPN API client (creates new if None)
            odds_client: TheOdds API client (creates new if None)
        """
        self.espn = espn_client or ESPNClient()
        self.odds = odds_client

    def ingest_teams(self, league: str, db: Session) -> int:
        """
        Fetch and store team data for a league.

        Args:
            league: League identifier (e.g., 'nba')
            db: Database session

        Returns:
            Number of teams ingested/updated
        """
        logger.info(f"Ingesting teams for {league.upper()}")

        # Fetch teams from ESPN
        teams_data = self.espn.get_teams(league)

        teams_count = 0

        for team_data in teams_data:
            try:
                # Extract team info
                team_info = team_data.get("team", team_data)

                external_id = team_info.get("id")
                if not external_id:
                    continue

                # Check if team exists
                existing_team = db.query(Team).filter(
                    and_(
                        Team.external_id == str(external_id),
                        Team.sport == league
                    )
                ).first()

                if existing_team:
                    # Update existing team
                    existing_team.name = team_info.get("name", existing_team.name)
                    existing_team.display_name = team_info.get(
                        "displayName",
                        existing_team.display_name
                    )
                    existing_team.abbreviation = team_info.get(
                        "abbreviation",
                        existing_team.abbreviation
                    )
                    existing_team.location = team_info.get("location")
                    existing_team.logo_url = team_info.get("logo")
                    existing_team.updated_at = datetime.now()
                else:
                    # Create new team
                    new_team = Team(
                        external_id=str(external_id),
                        sport=league.lower(),
                        league=league.upper(),
                        name=team_info.get("name"),
                        display_name=team_info.get("displayName"),
                        abbreviation=team_info.get("abbreviation"),
                        location=team_info.get("location"),
                        logo_url=team_info.get("logo"),
                    )
                    db.add(new_team)

                teams_count += 1

            except Exception as e:
                logger.error(f"Error ingesting team: {e}")
                continue

        db.commit()
        logger.info(f"Ingested {teams_count} teams for {league.upper()}")
        return teams_count

    def ingest_games(
        self,
        league: str,
        date: Optional[str] = None,
        db: Session = None
    ) -> int:
        """
        Fetch and store game data for a specific date.

        Args:
            league: League identifier
            date: Date in YYYYMMDD format (default: today)
            db: Database session

        Returns:
            Number of games ingested/updated
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        logger.info(f"Ingesting games for {league.upper()} on {date}")

        # Use context manager if no session provided
        if db is None:
            with get_db_context() as db:
                return self._ingest_games_with_db(league, date, db)
        else:
            return self._ingest_games_with_db(league, date, db)

    def _ingest_games_with_db(
        self,
        league: str,
        date: str,
        db: Session
    ) -> int:
        """Internal method to ingest games with a database session."""
        # Fetch games from ESPN
        scoreboard = self.espn.get_scoreboard(league, date)
        games_data = self.espn.parse_games(scoreboard)

        games_count = 0

        for game_data in games_data:
            try:
                external_id = game_data.get("game_id")
                if not external_id:
                    continue

                # Find or create home team
                home_team = self._get_or_create_team(
                    db=db,
                    external_id=game_data["home_team_id"],
                    name=game_data["home_team_name"],
                    abbreviation=game_data["home_team_abbr"],
                    sport=league,
                    logo_url=game_data.get("home_team_logo"),
                )

                # Find or create away team
                away_team = self._get_or_create_team(
                    db=db,
                    external_id=game_data["away_team_id"],
                    name=game_data["away_team_name"],
                    abbreviation=game_data["away_team_abbr"],
                    sport=league,
                    logo_url=game_data.get("away_team_logo"),
                )

                # Check if game exists
                existing_game = db.query(Game).filter(
                    Game.external_id == str(external_id)
                ).first()

                # Parse game date
                game_date = None
                if game_data.get("date"):
                    try:
                        game_date = datetime.fromisoformat(
                            game_data["date"].replace("Z", "+00:00")
                        )
                    except:
                        pass

                if existing_game:
                    # Update existing game
                    existing_game.status = game_data.get("status")
                    existing_game.home_score = game_data.get("home_score")
                    existing_game.away_score = game_data.get("away_score")
                    existing_game.venue = game_data.get("venue")
                    existing_game.attendance = game_data.get("attendance")
                    existing_game.updated_at = datetime.now()
                else:
                    # Create new game
                    new_game = Game(
                        external_id=str(external_id),
                        sport=league.lower(),
                        league=league.upper(),
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        game_date=game_date,
                        status=game_data.get("status"),
                        venue=game_data.get("venue"),
                        attendance=game_data.get("attendance"),
                        home_score=game_data.get("home_score"),
                        away_score=game_data.get("away_score"),
                        home_team_record=game_data.get("home_record"),
                        away_team_record=game_data.get("away_record"),
                        odds_data={"espn_odds": game_data.get("odds", [])},
                    )
                    db.add(new_game)

                games_count += 1

            except Exception as e:
                logger.error(f"Error ingesting game {game_data.get('game_id')}: {e}")
                continue

        db.commit()
        logger.info(f"Ingested {games_count} games for {league.upper()}")
        return games_count

    def ingest_odds(
        self,
        league: str,
        db: Session = None
    ) -> int:
        """
        Fetch and store betting odds for upcoming games.

        Args:
            league: League identifier
            db: Database session

        Returns:
            Number of games updated with odds
        """
        if not self.odds:
            logger.warning("No odds client configured, skipping odds ingestion")
            return 0

        logger.info(f"Ingesting odds for {league.upper()}")

        if db is None:
            with get_db_context() as db:
                return self._ingest_odds_with_db(league, db)
        else:
            return self._ingest_odds_with_db(league, db)

    def _ingest_odds_with_db(self, league: str, db: Session) -> int:
        """Internal method to ingest odds with database session."""
        try:
            # Fetch odds from TheOddsAPI
            odds_data = self.odds.get_odds(
                league,
                markets=["h2h", "spreads", "totals"]
            )
            parsed_odds = self.odds.parse_odds(odds_data)

            games_updated = 0

            for odds_game in parsed_odds:
                # Try to match with existing games
                home_team = odds_game["home_team"]
                away_team = odds_game["away_team"]
                commence_time = datetime.fromisoformat(
                    odds_game["commence_time"].replace("Z", "+00:00")
                )

                # Find matching game within a time window
                matching_games = db.query(Game).join(
                    Team, Game.home_team_id == Team.id
                ).filter(
                    and_(
                        Game.sport == league.lower(),
                        Game.game_date >= commence_time - timedelta(hours=2),
                        Game.game_date <= commence_time + timedelta(hours=2),
                    )
                ).all()

                # Try to find best match by team names
                matched_game = None
                for game in matching_games:
                    if (home_team.lower() in game.home_team.name.lower() or
                        away_team.lower() in game.away_team.name.lower()):
                        matched_game = game
                        break

                if matched_game:
                    # Update game with odds data
                    if matched_game.odds_data:
                        matched_game.odds_data["theodds"] = odds_game["bookmakers"]
                    else:
                        matched_game.odds_data = {"theodds": odds_game["bookmakers"]}

                    matched_game.updated_at = datetime.now()
                    games_updated += 1
                else:
                    logger.debug(
                        f"No matching game found for {away_team} @ {home_team}"
                    )

            db.commit()
            logger.info(f"Updated {games_updated} games with odds data")
            return games_updated

        except Exception as e:
            logger.error(f"Error ingesting odds: {e}")
            return 0

    def _get_or_create_team(
        self,
        db: Session,
        external_id: str,
        name: str,
        abbreviation: str,
        sport: str,
        logo_url: Optional[str] = None,
    ) -> Team:
        """Get existing team or create new one."""
        team = db.query(Team).filter(
            and_(
                Team.external_id == str(external_id),
                Team.sport == sport.lower()
            )
        ).first()

        if not team:
            team = Team(
                external_id=str(external_id),
                sport=sport.lower(),
                league=sport.upper(),
                name=name,
                display_name=name,
                abbreviation=abbreviation,
                logo_url=logo_url,
            )
            db.add(team)
            db.flush()  # Get the ID without committing

        return team

    def ingest_date_range(
        self,
        league: str,
        days_back: int = 7,
        days_forward: int = 7,
        include_odds: bool = True
    ) -> Dict[str, int]:
        """
        Ingest games and odds for a date range.

        Args:
            league: League identifier
            days_back: Number of days to look back
            days_forward: Number of days to look forward
            include_odds: Whether to fetch odds data

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(
            f"Ingesting {league.upper()} data for "
            f"{days_back} days back to {days_forward} days forward"
        )

        with get_db_context() as db:
            # Ingest teams first
            teams_count = self.ingest_teams(league, db)

            # Ingest games for date range
            total_games = 0
            start_date = datetime.now() - timedelta(days=days_back)
            end_date = datetime.now() + timedelta(days=days_forward)

            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")
                games_count = self._ingest_games_with_db(league, date_str, db)
                total_games += games_count
                current_date += timedelta(days=1)

            # Ingest odds if requested
            odds_count = 0
            if include_odds and self.odds:
                odds_count = self._ingest_odds_with_db(league, db)

        return {
            "teams": teams_count,
            "games": total_games,
            "games_with_odds": odds_count,
        }


# Convenience functions
def ingest_nba_today():
    """Quick function to ingest today's NBA games."""
    ingestion = DataIngestion()
    with get_db_context() as db:
        ingestion.ingest_teams("nba", db)
        games = ingestion.ingest_games("nba", db=db)
        return games


def ingest_nba_with_odds():
    """Ingest NBA games and odds."""
    from dotenv import load_dotenv
    import os

    load_dotenv()

    odds_client = None
    if os.getenv("THEODDS_API_KEY"):
        odds_client = TheOddsAPIClient()

    ingestion = DataIngestion(odds_client=odds_client)
    return ingestion.ingest_date_range("nba", days_back=1, days_forward=7)


if __name__ == "__main__":
    # Test ingestion
    from dotenv import load_dotenv
    load_dotenv()

    print("\n=== Testing Data Ingestion ===")
    print("\nIngesting NBA data...")

    stats = ingest_nba_with_odds()
    print(f"\nIngestion complete:")
    print(f"- Teams: {stats['teams']}")
    print(f"- Games: {stats['games']}")
    print(f"- Games with odds: {stats['games_with_odds']}")
