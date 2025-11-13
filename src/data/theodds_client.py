"""
TheOddsAPI Client for fetching betting odds and markets.
Official API documentation: https://the-odds-api.com/
"""

import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger
import os


class TheOddsAPIClient:
    """Client for fetching betting odds from TheOddsAPI."""

    BASE_URL = "https://api.the-odds-api.com/v4"

    # Sport key mappings
    SPORT_KEYS = {
        "nba": "basketball_nba",
        "nfl": "americanfootball_nfl",
        "ncaaf": "americanfootball_ncaaf",
        "mlb": "baseball_mlb",
        "nhl": "icehockey_nhl",
        "epl": "soccer_epl",  # English Premier League
        "mls": "soccer_usa_mls",
        "champions": "soccer_uefa_champs_league",
    }

    # Market types
    MARKETS = {
        "h2h": "Head to head (moneyline)",
        "spreads": "Point spreads",
        "totals": "Over/under totals",
        "outrights": "Futures/championship odds",
    }

    # Regions for odds
    REGIONS = ["us", "uk", "eu", "au"]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TheOddsAPI client.

        Args:
            api_key: API key for TheOddsAPI (or set THEODDS_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("THEODDS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TheOddsAPI key required. Set THEODDS_API_KEY env var "
                "or pass api_key parameter"
            )

        self.session = requests.Session()
        self.requests_remaining = None
        self.requests_used = None

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to TheOddsAPI.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response
        """
        url = f"{self.BASE_URL}/{endpoint}"

        # Add API key to params
        if params is None:
            params = {}
        params["apiKey"] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            # Track API usage from headers
            self.requests_remaining = response.headers.get("x-requests-remaining")
            self.requests_used = response.headers.get("x-requests-used")

            if self.requests_remaining:
                logger.info(f"API requests remaining: {self.requests_remaining}")

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"TheOddsAPI request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def get_sports(self, all_sports: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of available sports.

        Args:
            all_sports: Include all sports (True) or only in-season (False)

        Returns:
            List of available sports
        """
        params = {"all": str(all_sports).lower()}
        logger.info("Fetching available sports")
        return self._make_request("sports", params)

    def get_odds(
        self,
        sport: str,
        regions: Optional[List[str]] = None,
        markets: Optional[List[str]] = None,
        odds_format: str = "american",
        date_format: str = "iso",
    ) -> List[Dict[str, Any]]:
        """
        Get odds for upcoming games in a sport.

        Args:
            sport: Sport key (e.g., 'nba', 'nfl') or full key ('basketball_nba')
            regions: Regions for odds (default: ['us'])
            markets: Market types to include (default: ['h2h'])
            odds_format: Format for odds - 'american', 'decimal', 'hongkong'
            date_format: 'iso' or 'unix'

        Returns:
            List of games with odds
        """
        # Convert short sport key to full key if needed
        sport_key = self.SPORT_KEYS.get(sport.lower(), sport)

        if regions is None:
            regions = ["us"]
        if markets is None:
            markets = ["h2h"]

        params = {
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }

        logger.info(f"Fetching odds for {sport_key}")
        return self._make_request(f"sports/{sport_key}/odds", params)

    def get_historical_odds(
        self,
        sport: str,
        event_id: str,
        regions: Optional[List[str]] = None,
        markets: Optional[List[str]] = None,
        odds_format: str = "american",
        date_format: str = "iso",
    ) -> Dict[str, Any]:
        """
        Get historical odds for a specific event.

        Note: Historical odds may require a premium API plan.

        Args:
            sport: Sport key
            event_id: Event/game ID
            regions: Regions for odds
            markets: Market types
            odds_format: Format for odds
            date_format: Date format

        Returns:
            Historical odds data for the event
        """
        sport_key = self.SPORT_KEYS.get(sport.lower(), sport)

        if regions is None:
            regions = ["us"]
        if markets is None:
            markets = ["h2h", "spreads", "totals"]

        params = {
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }

        logger.info(f"Fetching historical odds for event {event_id}")
        return self._make_request(
            f"sports/{sport_key}/events/{event_id}/odds",
            params
        )

    def parse_odds(self, odds_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse odds data into simplified format.

        Args:
            odds_data: Raw odds response from API

        Returns:
            List of parsed game odds
        """
        parsed_games = []

        for event in odds_data:
            try:
                game = self._parse_single_event(event)
                if game:
                    parsed_games.append(game)
            except Exception as e:
                logger.error(f"Error parsing event {event.get('id')}: {e}")
                continue

        logger.info(f"Parsed {len(parsed_games)} events")
        return parsed_games

    def _parse_single_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single event's odds into simplified format."""
        game = {
            "event_id": event["id"],
            "sport_key": event["sport_key"],
            "sport_title": event["sport_title"],
            "commence_time": event["commence_time"],
            "home_team": event["home_team"],
            "away_team": event["away_team"],
            "bookmakers": [],
        }

        # Parse bookmaker odds
        for bookmaker in event.get("bookmakers", []):
            bookie_data = {
                "name": bookmaker["title"],
                "key": bookmaker["key"],
                "last_update": bookmaker["last_update"],
                "markets": {},
            }

            # Parse each market (h2h, spreads, totals)
            for market in bookmaker.get("markets", []):
                market_key = market["key"]
                bookie_data["markets"][market_key] = []

                for outcome in market.get("outcomes", []):
                    outcome_data = {
                        "name": outcome["name"],
                        "price": outcome.get("price"),  # Odds
                    }

                    # Add point/spread if available
                    if "point" in outcome:
                        outcome_data["point"] = outcome["point"]

                    bookie_data["markets"][market_key].append(outcome_data)

            game["bookmakers"].append(bookie_data)

        return game

    def get_best_odds(
        self,
        sport: str,
        team_name: str,
        market: str = "h2h"
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best odds for a specific team.

        Args:
            sport: Sport key
            team_name: Name of the team to find odds for
            market: Market type (h2h, spreads, totals)

        Returns:
            Best odds information or None if not found
        """
        odds_data = self.get_odds(sport, markets=[market])
        parsed_games = self.parse_odds(odds_data)

        best_odds = None
        best_price = None

        for game in parsed_games:
            if (game["home_team"] != team_name and
                game["away_team"] != team_name):
                continue

            for bookmaker in game["bookmakers"]:
                if market not in bookmaker["markets"]:
                    continue

                for outcome in bookmaker["markets"][market]:
                    if outcome["name"] == team_name:
                        price = outcome["price"]

                        # American odds: higher is better for underdogs (+),
                        # lower (closer to 0) is better for favorites (-)
                        if best_price is None or price > best_price:
                            best_price = price
                            best_odds = {
                                "team": team_name,
                                "bookmaker": bookmaker["name"],
                                "odds": price,
                                "market": market,
                                "game": f"{game['away_team']} @ {game['home_team']}",
                                "commence_time": game["commence_time"],
                            }

        return best_odds

    def get_consensus_odds(
        self,
        sport: str,
        market: str = "h2h"
    ) -> List[Dict[str, Any]]:
        """
        Calculate consensus (average) odds across all bookmakers.

        Args:
            sport: Sport key
            market: Market type

        Returns:
            List of games with consensus odds
        """
        odds_data = self.get_odds(sport, markets=[market])
        parsed_games = self.parse_odds(odds_data)

        consensus_games = []

        for game in parsed_games:
            # Collect all odds for each team
            team_odds = {}

            for bookmaker in game["bookmakers"]:
                if market not in bookmaker["markets"]:
                    continue

                for outcome in bookmaker["markets"][market]:
                    team = outcome["name"]
                    price = outcome["price"]

                    if team not in team_odds:
                        team_odds[team] = []
                    team_odds[team].append(price)

            # Calculate average odds
            consensus = {
                "game": f"{game['away_team']} @ {game['home_team']}",
                "event_id": game["event_id"],
                "commence_time": game["commence_time"],
                "teams": {},
            }

            for team, odds_list in team_odds.items():
                if odds_list:
                    consensus["teams"][team] = {
                        "avg_odds": sum(odds_list) / len(odds_list),
                        "min_odds": min(odds_list),
                        "max_odds": max(odds_list),
                        "num_books": len(odds_list),
                    }

            consensus_games.append(consensus)

        return consensus_games

    def american_to_probability(self, odds: float) -> float:
        """
        Convert American odds to implied probability.

        Args:
            odds: American odds (e.g., -110, +150)

        Returns:
            Implied probability (0 to 1)
        """
        if odds < 0:
            # Favorite
            return abs(odds) / (abs(odds) + 100)
        else:
            # Underdog
            return 100 / (odds + 100)

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current API usage statistics.

        Returns:
            Dictionary with usage information
        """
        return {
            "requests_used": self.requests_used,
            "requests_remaining": self.requests_remaining,
        }


# Convenience functions
def get_nba_odds() -> List[Dict[str, Any]]:
    """Quick helper to get current NBA odds."""
    client = TheOddsAPIClient()
    return client.get_odds("nba", markets=["h2h", "spreads", "totals"])


def get_nfl_odds() -> List[Dict[str, Any]]:
    """Quick helper to get current NFL odds."""
    client = TheOddsAPIClient()
    return client.get_odds("nfl", markets=["h2h", "spreads", "totals"])


if __name__ == "__main__":
    # Test the client (requires THEODDS_API_KEY environment variable)
    try:
        client = TheOddsAPIClient()

        print("\n=== Available Sports ===")
        sports = client.get_sports()
        for sport in sports[:5]:
            print(f"- {sport['title']} ({sport['key']})")

        print("\n=== NBA Odds Sample ===")
        nba_odds = client.get_odds("nba", markets=["h2h"])
        parsed = client.parse_odds(nba_odds)

        for game in parsed[:2]:
            print(f"\n{game['away_team']} @ {game['home_team']}")
            print(f"Commence: {game['commence_time']}")

            if game['bookmakers']:
                bookie = game['bookmakers'][0]
                print(f"Bookmaker: {bookie['name']}")
                if 'h2h' in bookie['markets']:
                    for outcome in bookie['markets']['h2h']:
                        prob = client.american_to_probability(outcome['price'])
                        print(f"  {outcome['name']}: {outcome['price']} "
                              f"(implied prob: {prob:.1%})")

        print(f"\n{client.get_usage_stats()}")

    except ValueError as e:
        print(f"Error: {e}")
        print("Set THEODDS_API_KEY environment variable to test.")
