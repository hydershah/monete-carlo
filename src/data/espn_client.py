"""
ESPN API Client for fetching sports data.
Uses the unofficial ESPN API endpoints.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger
import time


class ESPNClient:
    """Client for fetching data from ESPN's unofficial API."""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"

    # Sport and league mappings
    SPORT_MAPPINGS = {
        "nba": {"sport": "basketball", "league": "nba"},
        "nfl": {"sport": "football", "league": "nfl"},
        "mlb": {"sport": "baseball", "league": "mlb"},
        "nhl": {"sport": "hockey", "league": "nhl"},
        "soccer": {"sport": "soccer", "league": "eng.1"},  # Premier League
        "mls": {"sport": "soccer", "league": "usa.1"},
    }

    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize ESPN client.

        Args:
            rate_limit_delay: Delay between requests in seconds (default 0.5)
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0.0.0 Safari/537.36"
        })
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

    def _rate_limit(self):
        """Implement rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _get_sport_config(self, league: str) -> Dict[str, str]:
        """
        Get sport and league configuration.

        Args:
            league: League identifier (e.g., 'nba', 'nfl')

        Returns:
            Dictionary with sport and league keys
        """
        league_lower = league.lower()
        if league_lower not in self.SPORT_MAPPINGS:
            raise ValueError(
                f"Unsupported league: {league}. "
                f"Supported: {list(self.SPORT_MAPPINGS.keys())}"
            )
        return self.SPORT_MAPPINGS[league_lower]

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """
        Make HTTP request with error handling and rate limiting.

        Args:
            url: Full URL to request
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        self._rate_limit()

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"ESPN API request failed: {e}")
            raise

    def get_scoreboard(
        self,
        league: str,
        date: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get scoreboard for a specific league and date.

        Args:
            league: League identifier (e.g., 'nba', 'nfl')
            date: Date in YYYYMMDD format (default: today)
            limit: Maximum number of games to return

        Returns:
            Scoreboard data including games, scores, and teams
        """
        config = self._get_sport_config(league)
        url = (
            f"{self.BASE_URL}/{config['sport']}/{config['league']}/scoreboard"
        )

        params = {"limit": limit}
        if date:
            params["dates"] = date

        logger.info(f"Fetching {league.upper()} scoreboard for {date or 'today'}")
        return self._make_request(url, params)

    def get_team_info(self, league: str, team_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific team.

        Args:
            league: League identifier
            team_id: ESPN team ID

        Returns:
            Team information including roster, stats, etc.
        """
        config = self._get_sport_config(league)
        url = (
            f"{self.BASE_URL}/{config['sport']}/{config['league']}/"
            f"teams/{team_id}"
        )

        logger.info(f"Fetching team info for {league.upper()} team {team_id}")
        return self._make_request(url)

    def get_teams(self, league: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all teams in a league.

        Args:
            league: League identifier
            limit: Maximum number of teams to return

        Returns:
            List of team information
        """
        config = self._get_sport_config(league)
        url = f"{self.BASE_URL}/{config['sport']}/{config['league']}/teams"

        params = {"limit": limit}

        logger.info(f"Fetching all teams for {league.upper()}")
        data = self._make_request(url, params)

        # Extract teams from response
        if "sports" in data:
            teams = []
            for sport in data["sports"]:
                for league_data in sport.get("leagues", []):
                    teams.extend(league_data.get("teams", []))
            return teams
        elif "teams" in data:
            return data["teams"]
        else:
            return []

    def get_schedule(
        self,
        league: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get schedule for a league.

        Args:
            league: League identifier
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            limit: Maximum number of games

        Returns:
            Schedule data
        """
        config = self._get_sport_config(league)
        url = (
            f"{self.BASE_URL}/{config['sport']}/{config['league']}/scoreboard"
        )

        params = {"limit": limit}
        if start_date and end_date:
            # ESPN API uses dates parameter for range
            params["dates"] = f"{start_date}-{end_date}"

        logger.info(
            f"Fetching {league.upper()} schedule "
            f"from {start_date or 'now'} to {end_date or 'future'}"
        )
        return self._make_request(url, params)

    def parse_games(self, scoreboard_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse scoreboard data into simplified game format.

        Args:
            scoreboard_data: Raw scoreboard response from ESPN

        Returns:
            List of parsed game dictionaries
        """
        games = []

        if "events" not in scoreboard_data:
            logger.warning("No events found in scoreboard data")
            return games

        for event in scoreboard_data["events"]:
            try:
                game = self._parse_single_game(event)
                if game:
                    games.append(game)
            except Exception as e:
                logger.error(f"Error parsing game {event.get('id')}: {e}")
                continue

        logger.info(f"Parsed {len(games)} games")
        return games

    def _parse_single_game(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single game event into simplified format."""
        if len(event.get("competitions", [])) == 0:
            return None

        competition = event["competitions"][0]
        competitors = competition.get("competitors", [])

        if len(competitors) < 2:
            return None

        # Identify home and away teams
        home_team = next(
            (c for c in competitors if c.get("homeAway") == "home"),
            None
        )
        away_team = next(
            (c for c in competitors if c.get("homeAway") == "away"),
            None
        )

        if not home_team or not away_team:
            # Fallback to first two competitors
            home_team, away_team = competitors[0], competitors[1]

        # Parse game data
        game = {
            "game_id": event["id"],
            "date": event.get("date"),
            "name": event.get("name"),
            "short_name": event.get("shortName"),
            "status": competition["status"]["type"]["name"],
            "status_detail": competition["status"]["type"]["detail"],
            "completed": competition["status"]["type"]["completed"],

            # Home team
            "home_team_id": home_team["id"],
            "home_team_name": home_team["team"]["displayName"],
            "home_team_abbr": home_team["team"]["abbreviation"],
            "home_team_logo": home_team["team"].get("logo"),
            "home_score": int(home_team.get("score", 0)),
            "home_record": home_team.get("records", [{}])[0].get("summary", ""),

            # Away team
            "away_team_id": away_team["id"],
            "away_team_name": away_team["team"]["displayName"],
            "away_team_abbr": away_team["team"]["abbreviation"],
            "away_team_logo": away_team["team"].get("logo"),
            "away_score": int(away_team.get("score", 0)),
            "away_record": away_team.get("records", [{}])[0].get("summary", ""),

            # Additional info
            "venue": competition.get("venue", {}).get("fullName"),
            "attendance": competition.get("attendance"),
            "odds": competition.get("odds", []),
            "broadcast": competition.get("broadcasts", []),
        }

        return game

    def get_todays_games(self, league: str) -> List[Dict[str, Any]]:
        """
        Get today's games for a league.

        Args:
            league: League identifier

        Returns:
            List of parsed games for today
        """
        today = datetime.now().strftime("%Y%m%d")
        scoreboard = self.get_scoreboard(league, date=today)
        return self.parse_games(scoreboard)

    def get_games_date_range(
        self,
        league: str,
        days_back: int = 7,
        days_forward: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get games within a date range.

        Args:
            league: League identifier
            days_back: Number of days to look back
            days_forward: Number of days to look forward

        Returns:
            List of all games in the date range
        """
        all_games = []
        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now() + timedelta(days=days_forward)

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            try:
                scoreboard = self.get_scoreboard(league, date=date_str)
                games = self.parse_games(scoreboard)
                all_games.extend(games)
            except Exception as e:
                logger.error(f"Error fetching games for {date_str}: {e}")

            current_date += timedelta(days=1)

        logger.info(
            f"Fetched {len(all_games)} games from "
            f"{start_date.date()} to {end_date.date()}"
        )
        return all_games


# Convenience function
def get_nba_games_today() -> List[Dict[str, Any]]:
    """Quick helper to get today's NBA games."""
    client = ESPNClient()
    return client.get_todays_games("nba")


if __name__ == "__main__":
    # Test the client
    client = ESPNClient()

    # Test NBA scoreboard
    print("\n=== Testing NBA Today's Games ===")
    games = client.get_todays_games("nba")
    for game in games[:3]:  # Show first 3 games
        print(f"\n{game['name']}")
        print(f"Status: {game['status']}")
        print(f"Score: {game['away_team_abbr']} {game['away_score']} "
              f"@ {game['home_team_abbr']} {game['home_score']}")

    print(f"\nTotal games found: {len(games)}")
