"""
Redis Configuration for GameLens.ai Production Caching
Fast in-memory storage for current season stats and predictions
"""

import redis
import json
import os
from typing import Any, Optional, Dict
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis cache manager for GameLens.ai
    Handles team stats, predictions, and live data caching
    """

    # Cache TTL (Time-To-Live) constants
    TTL_TEAM_STATS = 6 * 3600          # 6 hours (updated by background job)
    TTL_PREDICTION = 15 * 60            # 15 minutes (odds change frequently)
    TTL_LIVE_ODDS = 5 * 60              # 5 minutes (during active betting hours)
    TTL_GAME_INFO = 24 * 3600           # 24 hours (schedule info)
    TTL_MODEL_WEIGHTS = 24 * 3600       # 24 hours (updated daily)

    # Cache key prefixes
    PREFIX_TEAM_STATS = "team_stats"
    PREFIX_PREDICTION = "prediction"
    PREFIX_ODDS = "odds"
    PREFIX_GAME = "game"
    PREFIX_MODEL_WEIGHTS = "model_weights"
    PREFIX_ELO = "elo_rating"

    def __init__(self, host: str = None, port: int = None, db: int = 0, password: str = None):
        """
        Initialize Redis connection

        Args:
            host: Redis host (default from env or localhost)
            port: Redis port (default from env or 6379)
            db: Redis database number (default 0)
            password: Redis password (optional)
        """
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = port or int(os.getenv('REDIS_PORT', 6379))
        self.db = db
        self.password = password or os.getenv('REDIS_PASSWORD')

        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,  # Auto-decode bytes to strings
                socket_timeout=5,
                socket_connect_timeout=5
            )

            # Test connection
            self.client.ping()
            logger.info(f"✅ Redis connected: {self.host}:{self.port} (db={self.db})")

        except redis.ConnectionError as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.warning("⚠️  Running without cache - performance will be degraded")
            self.client = None

    # ==================== TEAM STATS CACHING ====================

    def cache_team_stats(self, team_id: str, league: str, stats: Dict[str, Any]) -> bool:
        """
        Cache team statistics (updated every 6 hours by background job)

        Args:
            team_id: Team identifier
            league: League (NBA, NFL, etc.)
            stats: Dictionary of team statistics

        Returns:
            True if cached successfully
        """
        if not self.client:
            return False

        key = f"{self.PREFIX_TEAM_STATS}:{league}:{team_id}"

        try:
            self.client.setex(
                key,
                self.TTL_TEAM_STATS,
                json.dumps(stats)
            )
            logger.debug(f"Cached team stats: {key}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache team stats: {e}")
            return False

    def get_team_stats(self, team_id: str, league: str) -> Optional[Dict[str, Any]]:
        """
        Get cached team statistics

        Args:
            team_id: Team identifier
            league: League (NBA, NFL, etc.)

        Returns:
            Dictionary of team stats or None if not found
        """
        if not self.client:
            return None

        key = f"{self.PREFIX_TEAM_STATS}:{league}:{team_id}"

        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
            return None

        except Exception as e:
            logger.error(f"Failed to get team stats: {e}")
            return None

    # ==================== PREDICTION CACHING ====================

    def cache_prediction(self, home_team: str, away_team: str, spread: float,
                        total: float, prediction: Dict[str, Any]) -> bool:
        """
        Cache game prediction (15 minute TTL)

        Args:
            home_team: Home team ID
            away_team: Away team ID
            spread: Point spread
            total: Over/under total
            prediction: Prediction dictionary

        Returns:
            True if cached successfully
        """
        if not self.client:
            return False

        # Create unique key including odds (odds change = new prediction)
        key = f"{self.PREFIX_PREDICTION}:{home_team}:{away_team}:{spread}:{total}"

        try:
            self.client.setex(
                key,
                self.TTL_PREDICTION,
                json.dumps(prediction)
            )
            logger.debug(f"Cached prediction: {key}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache prediction: {e}")
            return False

    def get_prediction(self, home_team: str, away_team: str, spread: float,
                      total: float) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction

        Args:
            home_team: Home team ID
            away_team: Away team ID
            spread: Point spread
            total: Over/under total

        Returns:
            Prediction dictionary or None if not found
        """
        if not self.client:
            return None

        key = f"{self.PREFIX_PREDICTION}:{home_team}:{away_team}:{spread}:{total}"

        try:
            data = self.client.get(key)
            if data:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(data)

            logger.debug(f"Cache MISS: {key}")
            return None

        except Exception as e:
            logger.error(f"Failed to get prediction: {e}")
            return None

    # ==================== ODDS CACHING ====================

    def cache_odds(self, game_id: str, odds_data: Dict[str, Any]) -> bool:
        """
        Cache current odds for a game

        Args:
            game_id: Game identifier
            odds_data: Dictionary of odds information

        Returns:
            True if cached successfully
        """
        if not self.client:
            return False

        key = f"{self.PREFIX_ODDS}:{game_id}"

        try:
            self.client.setex(
                key,
                self.TTL_LIVE_ODDS,
                json.dumps(odds_data)
            )
            return True

        except Exception as e:
            logger.error(f"Failed to cache odds: {e}")
            return False

    def get_odds(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get cached odds for a game"""
        if not self.client:
            return None

        key = f"{self.PREFIX_ODDS}:{game_id}"

        try:
            data = self.client.get(key)
            return json.loads(data) if data else None

        except Exception as e:
            logger.error(f"Failed to get odds: {e}")
            return None

    # ==================== MODEL WEIGHTS CACHING ====================

    def cache_model_weights(self, league: str, weights: Dict[str, float]) -> bool:
        """
        Cache dynamic model weights for ensemble

        Args:
            league: League (NBA, NFL, etc.)
            weights: Dictionary of model weights

        Returns:
            True if cached successfully
        """
        if not self.client:
            return False

        key = f"{self.PREFIX_MODEL_WEIGHTS}:{league}"

        try:
            self.client.setex(
                key,
                self.TTL_MODEL_WEIGHTS,
                json.dumps(weights)
            )
            logger.info(f"Cached model weights for {league}: {weights}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache model weights: {e}")
            return False

    def get_model_weights(self, league: str) -> Optional[Dict[str, float]]:
        """Get cached model weights for ensemble"""
        if not self.client:
            return None

        key = f"{self.PREFIX_MODEL_WEIGHTS}:{league}"

        try:
            data = self.client.get(key)
            return json.loads(data) if data else None

        except Exception as e:
            logger.error(f"Failed to get model weights: {e}")
            return None

    # ==================== ELO RATING CACHING ====================

    def cache_elo_rating(self, team_id: str, league: str, rating: float) -> bool:
        """Cache current Elo rating for a team"""
        if not self.client:
            return False

        key = f"{self.PREFIX_ELO}:{league}:{team_id}"

        try:
            self.client.setex(
                key,
                self.TTL_TEAM_STATS,
                str(rating)
            )
            return True

        except Exception as e:
            logger.error(f"Failed to cache Elo rating: {e}")
            return False

    def get_elo_rating(self, team_id: str, league: str) -> Optional[float]:
        """Get cached Elo rating for a team"""
        if not self.client:
            return None

        key = f"{self.PREFIX_ELO}:{league}:{team_id}"

        try:
            data = self.client.get(key)
            return float(data) if data else None

        except Exception as e:
            logger.error(f"Failed to get Elo rating: {e}")
            return None

    # ==================== UTILITY METHODS ====================

    def clear_cache(self, pattern: str = None) -> int:
        """
        Clear cache entries matching pattern

        Args:
            pattern: Redis key pattern (e.g., "prediction:*")
                    If None, clears entire cache

        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0

        try:
            if pattern:
                keys = self.client.keys(pattern)
            else:
                keys = self.client.keys('*')

            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self.client:
            return {'status': 'disconnected'}

        try:
            info = self.client.info('stats')

            return {
                'status': 'connected',
                'total_keys': self.client.dbsize(),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(
                    info.get('keyspace_hits', 0),
                    info.get('keyspace_misses', 0)
                ),
                'memory_used': info.get('used_memory_human', 'unknown'),
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'status': 'error', 'error': str(e)}

    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage"""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0

    def warmup_cache(self, team_stats: Dict[str, Dict], model_weights: Dict[str, Dict]) -> None:
        """
        Pre-warm cache with team stats and model weights
        Called by background job before game day

        Args:
            team_stats: Dictionary of {team_id: stats_dict}
            model_weights: Dictionary of {league: weights_dict}
        """
        logger.info("Warming up Redis cache...")

        # Cache all team stats
        for key, stats in team_stats.items():
            team_id, league = key.split(':')
            self.cache_team_stats(team_id, league, stats)

        # Cache all model weights
        for league, weights in model_weights.items():
            self.cache_model_weights(league, weights)

        logger.info(f"✅ Cache warmed: {len(team_stats)} teams, {len(model_weights)} leagues")


# Global cache instance
_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Get global Redis cache instance (singleton)"""
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = RedisCache()

    return _cache_instance


if __name__ == "__main__":
    # Test Redis connection
    print("Testing Redis connection...")

    cache = RedisCache()

    # Test stats
    print(f"\nCache stats: {cache.get_cache_stats()}")

    # Test team stats caching
    test_stats = {
        'elo_rating': 1650,
        'ppg': 115.3,
        'papg': 108.2,
        'win_pct': 0.650
    }

    cache.cache_team_stats('lakers', 'NBA', test_stats)
    retrieved = cache.get_team_stats('lakers', 'NBA')

    print(f"\nTest cache:")
    print(f"  Stored: {test_stats}")
    print(f"  Retrieved: {retrieved}")
    print(f"  ✅ Match: {test_stats == retrieved}")

    # Clean up
    cache.clear_cache("team_stats:*")
    print(f"\n✅ Redis test complete!")
