"""
Goalserve API Client for Sports Data Integration
Provides access to live scores, odds, injuries, stats for NFL, NBA, MLB, NHL, Soccer, etc.
"""

import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
import os
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

@dataclass
class GoalserveGame:
    """Standardized game data from Goalserve"""
    game_id: str
    home_team: str
    away_team: str
    home_score: Optional[int]
    away_score: Optional[int]
    game_time: datetime
    status: str
    league: str
    spread: Optional[float] = None
    total: Optional[float] = None
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    venue: Optional[str] = None


class GoalserveClient:
    """
    Client for Goalserve API
    Provides comprehensive sports data including scores, odds, injuries, and stats
    """

    BASE_URL = "http://www.goalserve.com/getfeed"

    # Sport-specific endpoints
    ENDPOINTS = {
        # NFL
        'nfl_scores': '/football/nfl-scores',
        'nfl_scores_history': '/football/nfl-scores?date={date}',
        'nfl_playbyplay': '/football/nfl-playbyplay-scores',
        'nfl_schedule': '/football/nfl-shedule',
        'nfl_standings': '/football/nfl-standings',
        'nfl_odds': '/football/nfl-shedule?date1={date1}&date2={date2}&showodds=1',
        'nfl_injuries': '/football/{team_id}_injuries',
        'nfl_roster': '/football/{team_id}_rosters',
        'nfl_stats': '/football/{team_id}_player_stats',

        # NCAA Football
        'ncaaf_scores': '/football/fbs-scores',
        'ncaaf_scores_history': '/football/fbs-scores?date={date}',
        'ncaaf_playbyplay': '/football/fbs-playbyplay-scores',
        'ncaaf_schedule': '/football/fbs-shedule',
        'ncaaf_standings': '/football/fbs-standings',
        'ncaaf_odds': '/football/fbs-shedule?date1={date1}&date2={date2}&showodds=1',

        # NBA
        'nba_scores': '/bsktbl/nba-scores',
        'nba_scores_history': '/bsktbl/nba-scores?date={date}',
        'nba_playbyplay': '/bsktbl/nba-playbyplay',
        'nba_schedule': '/bsktbl/nba-shedule',
        'nba_standings': '/bsktbl/nba-standings',
        'nba_odds': '/bsktbl/nba-shedule?date1={date1}&date2={date2}&showodds=1',
        'nba_roster': '/bsktbl/{team_id}_rosters',
        'nba_stats': '/bsktbl/{team_id}_stats',
        'nba_team_stats': '/bsktbl/{team_id}_team_stats',
        'nba_injuries': '/bsktbl/{team_id}_injuries',

        # NCAA Basketball
        'ncaab_scores': '/bsktbl/ncaa-scores',
        'ncaab_scores_history': '/bsktbl/ncaa-scores?date={date}',
        'ncaab_playbyplay': '/bsktbl/ncaa-playbyplay',
        'ncaab_schedule': '/bsktbl/ncaa-shedule',
        'ncaab_standings': '/bsktbl/ncaa-standings',
        'ncaab_odds': '/bsktbl/ncaa-shedule?date1={date1}&date2={date2}&showodds=1',

        # MLB
        'mlb_scores': '/baseball/mlb-scores',
        'mlb_scores_history': '/baseball/usa?date={date}',
        'mlb_playbyplay': '/baseball/mlb-playbyplay',
        'mlb_schedule': '/baseball/mlb_shedule',
        'mlb_standings': '/baseball/mlb_standings',
        'mlb_odds': '/baseball/mlb_shedule?date1={date1}&date2={date2}&showodds=true',
        'mlb_roster': '/baseball/{team_id}_rosters',
        'mlb_stats': '/baseball/{team_id}_stats',
        'mlb_team_stats': '/baseball/{team_id}_team_stats',
        'mlb_injuries': '/baseball/{team_id}_injuries',

        # NHL
        'nhl_scores': '/hockey/nhl-scores',
        'nhl_scores_history': '/hockey/nhl-scores?date={date}',
        'nhl_schedule': '/hockey/nhl-shedule',
        'nhl_standings': '/hockey/nhl-standings',
        'nhl_odds': '/hockey/nhl-shedule?date1={date1}&date2={date2}&showodds=1',
        'nhl_roster': '/hockey/{team_id}_rosters',
        'nhl_injuries': '/hockey/{team_id}_injuries',
        'nhl_stats': '/hockey/{team_id}_stats',
        'nhl_team_stats': '/hockey/{team_id}_team_stats',

        # Soccer MLS
        'mls_fixtures': '/soccerfixtures/usa/mls',
        'mls_scores': '/commentaries/1440.xml',
        'mls_standings': '/standings/usa.xml',
        'mls_odds': '/getodds/soccer?cat=usa',

        # Soccer EPL
        'epl_fixtures': '/soccerfixtures/england/premierleague',
        'epl_scores': '/commentaries/1204.xml',
        'epl_standings': '/standings/england',
        'epl_odds': '/getodds/soccer?cat=england',
    }

    # Team ID mappings for common teams
    TEAM_IDS = {
        'nfl': {
            'Arizona Cardinals': '1691',
            'Atlanta Falcons': '1692',
            'Baltimore Ravens': '1693',
            'Buffalo Bills': '1694',
            'Carolina Panthers': '1695',
            'Chicago Bears': '1696',
            'Cincinnati Bengals': '1697',
            'Cleveland Browns': '1698',
            'Dallas Cowboys': '1699',
            'Denver Broncos': '1700',
            'Detroit Lions': '1701',
            'Green Bay Packers': '1702',
            'Houston Texans': '1703',
            'Indianapolis Colts': '1704',
            'Jacksonville Jaguars': '1705',
            'Kansas City Chiefs': '1706',
            'Las Vegas Raiders': '1707',
            'Los Angeles Chargers': '1708',
            'Los Angeles Rams': '1709',
            'Miami Dolphins': '1710',
            'Minnesota Vikings': '1711',
            'New England Patriots': '1712',
            'New Orleans Saints': '1713',
            'New York Giants': '1714',
            'New York Jets': '1715',
            'Philadelphia Eagles': '1716',
            'Pittsburgh Steelers': '1717',
            'San Francisco 49ers': '1718',
            'Seattle Seahawks': '1719',
            'Tampa Bay Buccaneers': '1720',
            'Tennessee Titans': '1721',
            'Washington Commanders': '1722'
        },
        'nba': {
            'Atlanta Hawks': '1193',
            'Boston Celtics': '1194',
            'Brooklyn Nets': '1195',
            'Charlotte Hornets': '1196',
            'Chicago Bulls': '1197',
            'Cleveland Cavaliers': '1198',
            'Dallas Mavericks': '1199',
            'Denver Nuggets': '1200',
            'Detroit Pistons': '1201',
            'Golden State Warriors': '1202',
            'Houston Rockets': '1203',
            'Indiana Pacers': '1204',
            'LA Clippers': '1205',
            'Los Angeles Lakers': '1206',
            'Memphis Grizzlies': '1207',
            'Miami Heat': '1208',
            'Milwaukee Bucks': '1209',
            'Minnesota Timberwolves': '1210',
            'New Orleans Pelicans': '1211',
            'New York Knicks': '1212',
            'Oklahoma City Thunder': '1213',
            'Orlando Magic': '1214',
            'Philadelphia 76ers': '1215',
            'Phoenix Suns': '1216',
            'Portland Trail Blazers': '1217',
            'Sacramento Kings': '1218',
            'San Antonio Spurs': '1219',
            'Toronto Raptors': '1220',
            'Utah Jazz': '1221',
            'Washington Wizards': '1222'
        }
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Goalserve client

        Args:
            api_key: Goalserve API key (or set GOALSERVE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('GOALSERVE_API_KEY', '94d5c09b345643d75bcf08de20cee058')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GameLens.ai/1.0'
        })

    def _build_url(self, endpoint: str, json_format: bool = True) -> str:
        """Build full URL with API key and JSON format"""
        url = f"{self.BASE_URL}/{self.api_key}{endpoint}"
        if json_format:
            url += "?json=1" if "?" not in url else "&json=1"
        return url

    def _make_request(self, endpoint: str, json_format: bool = True) -> Optional[Any]:
        """Make API request with error handling"""
        try:
            url = self._build_url(endpoint, json_format)
            logger.debug(f"Requesting: {url}")

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            if json_format:
                return response.json()
            else:
                return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None

    def get_scores(self, sport: str, date: Optional[datetime] = None) -> List[GoalserveGame]:
        """
        Get live scores or historical scores for a sport

        Args:
            sport: Sport type (nfl, nba, mlb, nhl, ncaaf, ncaab)
            date: Optional date for historical scores

        Returns:
            List of GoalserveGame objects
        """
        if date:
            date_str = date.strftime("%d.%m.%Y")
            endpoint_key = f"{sport}_scores_history"
            endpoint = self.ENDPOINTS[endpoint_key].format(date=date_str)
        else:
            endpoint = self.ENDPOINTS[f"{sport}_scores"]

        data = self._make_request(endpoint)

        if not data:
            return []

        return self._parse_scores(data, sport)

    def get_odds(self, sport: str, date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get betting odds for games

        Args:
            sport: Sport type
            date: Date for odds (defaults to today)

        Returns:
            List of games with odds
        """
        if not date:
            date = datetime.now()

        date_str = date.strftime("%d.%m.%Y")
        endpoint_key = f"{sport}_odds"

        if endpoint_key not in self.ENDPOINTS:
            logger.warning(f"Odds endpoint not available for {sport}")
            return []

        endpoint = self.ENDPOINTS[endpoint_key].format(
            date1=date_str,
            date2=date_str
        )

        data = self._make_request(endpoint)
        return self._parse_odds(data, sport) if data else []

    def get_injuries(self, sport: str, team_name: str) -> List[Dict[str, Any]]:
        """
        Get injury report for a team

        Args:
            sport: Sport type
            team_name: Team name

        Returns:
            List of injured players with details
        """
        # Get team ID
        team_id = self._get_team_id(sport, team_name)
        if not team_id:
            logger.warning(f"Team ID not found for {team_name}")
            return []

        endpoint_key = f"{sport}_injuries"
        if endpoint_key not in self.ENDPOINTS:
            logger.warning(f"Injuries endpoint not available for {sport}")
            return []

        endpoint = self.ENDPOINTS[endpoint_key].format(team_id=team_id)
        data = self._make_request(endpoint)

        return self._parse_injuries(data) if data else []

    def get_team_stats(self, sport: str, team_name: str) -> Dict[str, Any]:
        """
        Get team statistics

        Args:
            sport: Sport type
            team_name: Team name

        Returns:
            Dictionary of team statistics
        """
        team_id = self._get_team_id(sport, team_name)
        if not team_id:
            return {}

        endpoint_key = f"{sport}_team_stats"
        if endpoint_key not in self.ENDPOINTS:
            # Try regular stats endpoint
            endpoint_key = f"{sport}_stats"

        if endpoint_key not in self.ENDPOINTS:
            logger.warning(f"Stats endpoint not available for {sport}")
            return {}

        endpoint = self.ENDPOINTS[endpoint_key].format(team_id=team_id)
        data = self._make_request(endpoint)

        return self._parse_team_stats(data) if data else {}

    def get_standings(self, sport: str) -> Dict[str, Any]:
        """
        Get league standings

        Args:
            sport: Sport type

        Returns:
            Dictionary with conference/division standings
        """
        endpoint_key = f"{sport}_standings"
        if endpoint_key not in self.ENDPOINTS:
            logger.warning(f"Standings endpoint not available for {sport}")
            return {}

        endpoint = self.ENDPOINTS[endpoint_key]
        data = self._make_request(endpoint)

        return self._parse_standings(data, sport) if data else {}

    def get_schedule(self, sport: str, start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> List[GoalserveGame]:
        """
        Get upcoming schedule

        Args:
            sport: Sport type
            start_date: Start date for schedule
            end_date: End date for schedule

        Returns:
            List of scheduled games
        """
        endpoint = self.ENDPOINTS.get(f"{sport}_schedule")
        if not endpoint:
            logger.warning(f"Schedule endpoint not available for {sport}")
            return []

        # Add date parameters if provided
        params = []
        if start_date:
            params.append(f"date1={start_date.strftime('%d.%m.%Y')}")
        if end_date:
            params.append(f"date2={end_date.strftime('%d.%m.%Y')}")

        if params:
            endpoint += "?" + "&".join(params)

        data = self._make_request(endpoint)
        return self._parse_schedule(data, sport) if data else []

    def _get_team_id(self, sport: str, team_name: str) -> Optional[str]:
        """Get team ID from team name"""
        if sport in self.TEAM_IDS:
            return self.TEAM_IDS[sport].get(team_name)
        return None

    def _parse_scores(self, data: Any, sport: str) -> List[GoalserveGame]:
        """Parse scores data into standardized format"""
        games = []

        try:
            # The structure varies by sport, but generally follows similar patterns
            # This is a simplified parser - expand based on actual API response
            if isinstance(data, dict):
                # Navigate to games list (structure varies by sport)
                if 'scores' in data:
                    scores_data = data['scores']
                elif 'games' in data:
                    scores_data = data['games']
                else:
                    scores_data = data

                # Parse each game
                if isinstance(scores_data, dict):
                    for game_id, game_data in scores_data.items():
                        if isinstance(game_data, dict):
                            game = self._parse_single_game(game_data, game_id, sport)
                            if game:
                                games.append(game)
                elif isinstance(scores_data, list):
                    for game_data in scores_data:
                        game = self._parse_single_game(game_data, None, sport)
                        if game:
                            games.append(game)

        except Exception as e:
            logger.error(f"Error parsing scores: {e}")

        return games

    def _parse_single_game(self, game_data: Dict, game_id: Optional[str], sport: str) -> Optional[GoalserveGame]:
        """Parse single game data"""
        try:
            # Extract common fields (adjust based on actual API structure)
            home_team = game_data.get('hometeam', {}).get('name', '')
            away_team = game_data.get('awayteam', {}).get('name', '')

            if not home_team or not away_team:
                return None

            # Parse scores
            home_score = None
            away_score = None
            if 'score' in game_data:
                home_score = int(game_data['score'].get('home', 0))
                away_score = int(game_data['score'].get('away', 0))

            # Parse time
            game_time = datetime.now()  # Default
            if 'time' in game_data:
                try:
                    game_time = datetime.strptime(game_data['time'], "%Y-%m-%d %H:%M:%S")
                except:
                    pass

            # Parse status
            status = game_data.get('status', 'scheduled')

            return GoalserveGame(
                game_id=game_id or game_data.get('id', ''),
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                game_time=game_time,
                status=status,
                league=sport.upper(),
                venue=game_data.get('venue', '')
            )

        except Exception as e:
            logger.error(f"Error parsing game: {e}")
            return None

    def _parse_odds(self, data: Any, sport: str) -> List[Dict[str, Any]]:
        """Parse odds data"""
        odds_list = []

        try:
            # Parse based on actual API structure
            # This is a placeholder implementation
            if isinstance(data, dict) and 'odds' in data:
                for game_odds in data['odds']:
                    odds_list.append({
                        'home_team': game_odds.get('home_team'),
                        'away_team': game_odds.get('away_team'),
                        'spread': float(game_odds.get('spread', 0)),
                        'total': float(game_odds.get('total', 0)),
                        'home_ml': float(game_odds.get('home_ml', 0)),
                        'away_ml': float(game_odds.get('away_ml', 0))
                    })
        except Exception as e:
            logger.error(f"Error parsing odds: {e}")

        return odds_list

    def _parse_injuries(self, data: Any) -> List[Dict[str, Any]]:
        """Parse injury data"""
        injuries = []

        try:
            if isinstance(data, dict) and 'injuries' in data:
                for player in data['injuries']:
                    injuries.append({
                        'player_name': player.get('name'),
                        'position': player.get('position'),
                        'status': player.get('status'),
                        'injury': player.get('injury'),
                        'return_date': player.get('return_date')
                    })
        except Exception as e:
            logger.error(f"Error parsing injuries: {e}")

        return injuries

    def _parse_team_stats(self, data: Any) -> Dict[str, Any]:
        """Parse team statistics"""
        stats = {}

        try:
            if isinstance(data, dict):
                # Extract relevant statistics based on sport
                stats = {
                    'wins': data.get('wins', 0),
                    'losses': data.get('losses', 0),
                    'points_for': data.get('points_for', 0),
                    'points_against': data.get('points_against', 0),
                    'offensive_rating': data.get('offensive_rating', 0),
                    'defensive_rating': data.get('defensive_rating', 0)
                }
        except Exception as e:
            logger.error(f"Error parsing team stats: {e}")

        return stats

    def _parse_standings(self, data: Any, sport: str) -> Dict[str, Any]:
        """Parse standings data"""
        standings = {}

        try:
            if isinstance(data, dict):
                # Structure varies by sport
                standings = data
        except Exception as e:
            logger.error(f"Error parsing standings: {e}")

        return standings

    def _parse_schedule(self, data: Any, sport: str) -> List[GoalserveGame]:
        """Parse schedule data"""
        # Similar to parse_scores but for future games
        return self._parse_scores(data, sport)


if __name__ == "__main__":
    # Test the Goalserve client
    client = GoalserveClient()

    # Test getting NBA scores
    print("Testing NBA scores...")
    nba_games = client.get_scores('nba')
    if nba_games:
        print(f"Found {len(nba_games)} NBA games")
        for game in nba_games[:3]:  # Show first 3
            print(f"  {game.away_team} @ {game.home_team}: "
                  f"{game.away_score or 0}-{game.home_score or 0}")

    # Test getting NFL odds
    print("\nTesting NFL odds...")
    nfl_odds = client.get_odds('nfl')
    if nfl_odds:
        print(f"Found {len(nfl_odds)} NFL games with odds")
        for game in nfl_odds[:3]:
            print(f"  {game.get('away_team')} @ {game.get('home_team')}: "
                  f"Spread {game.get('spread')}, Total {game.get('total')}")

    # Test getting injuries
    print("\nTesting Lakers injuries...")
    lakers_injuries = client.get_injuries('nba', 'Los Angeles Lakers')
    if lakers_injuries:
        print(f"Found {len(lakers_injuries)} injured players")
        for player in lakers_injuries:
            print(f"  {player.get('player_name')}: {player.get('injury')} "
                  f"({player.get('status')})")

    print("\nGoalserve client test completed!")