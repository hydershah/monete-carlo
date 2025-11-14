"""
Closing Line Value (CLV) Tracking System
=========================================
Research shows 79.7% of bettors beating closing lines are profitable long-term.
CLV is the definitive indicator of long-term profitability.

This module tracks and analyzes CLV across all predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LineMovement:
    """Track line movement from opening to closing"""
    game_id: str
    bet_type: str  # 'spread', 'moneyline', 'total'
    opening_line: float
    bet_line: float  # Line when bet was placed
    closing_line: float
    line_movement: float  # Total movement from open to close
    timestamp_open: datetime
    timestamp_bet: datetime
    timestamp_close: datetime


@dataclass
class CLVResult:
    """Result of CLV calculation"""
    clv_points: float  # For spreads/totals
    clv_percentage: float  # Percentage edge
    beat_closing: bool  # True if positive CLV
    confidence_level: str  # 'HIGH', 'MEDIUM', 'LOW'


class CLVTracker:
    """
    Track and analyze Closing Line Value
    Research shows CLV predicts long-term success better than win/loss
    """

    def __init__(self):
        """Initialize CLV tracker"""
        self.line_history = {}  # Store line movements
        self.clv_records = []  # Store all CLV calculations
        self.performance_by_clv = {
            'positive_clv': {'bets': 0, 'wins': 0, 'profit': 0.0},
            'negative_clv': {'bets': 0, 'wins': 0, 'profit': 0.0}
        }

    def calculate_clv(self,
                     bet_odds: float,
                     closing_odds: float,
                     bet_type: str) -> CLVResult:
        """
        Calculate CLV for different bet types

        Spread/Total: Simple difference in lines
        Moneyline: Convert to implied probability and compare

        Args:
            bet_odds: Odds when bet was placed
            closing_odds: Closing odds
            bet_type: 'spread', 'total', or 'moneyline'

        Returns:
            CLVResult with all metrics
        """
        if bet_type in ['spread', 'total']:
            # For spreads/totals: CLV in points
            clv_points = bet_odds - closing_odds

            # Convert to percentage advantage
            # Rule of thumb: 0.5 point = ~2% edge
            clv_percentage = clv_points * 2.0

            # Determine confidence based on CLV size
            if abs(clv_percentage) > 4:
                confidence = 'HIGH'
            elif abs(clv_percentage) > 2:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

            return CLVResult(
                clv_points=clv_points,
                clv_percentage=clv_percentage,
                beat_closing=clv_points > 0,
                confidence_level=confidence
            )

        elif bet_type == 'moneyline':
            # Convert American odds to implied probability
            bet_prob = self._odds_to_probability(bet_odds)
            close_prob = self._odds_to_probability(closing_odds)

            # CLV in probability space
            clv_probability = bet_prob - close_prob
            clv_percentage = ((bet_prob - close_prob) / close_prob) * 100 if close_prob > 0 else 0

            # Determine confidence
            if abs(clv_percentage) > 5:
                confidence = 'HIGH'
            elif abs(clv_percentage) > 2.5:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

            return CLVResult(
                clv_points=clv_probability * 100,  # Convert to percentage points
                clv_percentage=clv_percentage,
                beat_closing=bet_prob < close_prob,  # Lower prob = better odds
                confidence_level=confidence
            )

        else:
            raise ValueError(f"Unknown bet type: {bet_type}")

    def _odds_to_probability(self, american_odds: float) -> float:
        """
        Convert American odds to implied probability

        Args:
            american_odds: American format odds (-110, +150, etc.)

        Returns:
            Implied probability (0-1)
        """
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    def _probability_to_odds(self, probability: float) -> float:
        """
        Convert probability to American odds

        Args:
            probability: Win probability (0-1)

        Returns:
            American format odds
        """
        if probability >= 0.5:
            # Favorite
            return -(probability / (1 - probability)) * 100
        else:
            # Underdog
            return ((1 - probability) / probability) * 100

    def track_line_movement(self,
                           game_id: str,
                           bet_type: str,
                           opening_line: float,
                           bet_line: float,
                           closing_line: float,
                           timestamps: Optional[Dict[str, datetime]] = None) -> LineMovement:
        """
        Track line movement from opening to closing

        Args:
            game_id: Unique game identifier
            bet_type: Type of bet
            opening_line: Opening line
            bet_line: Line when bet was placed
            closing_line: Closing line
            timestamps: Optional timestamps for each line

        Returns:
            LineMovement object
        """
        now = datetime.now()
        timestamps = timestamps or {}

        movement = LineMovement(
            game_id=game_id,
            bet_type=bet_type,
            opening_line=opening_line,
            bet_line=bet_line,
            closing_line=closing_line,
            line_movement=closing_line - opening_line,
            timestamp_open=timestamps.get('open', now - timedelta(hours=24)),
            timestamp_bet=timestamps.get('bet', now - timedelta(hours=12)),
            timestamp_close=timestamps.get('close', now)
        )

        # Store in history
        if game_id not in self.line_history:
            self.line_history[game_id] = {}
        self.line_history[game_id][bet_type] = movement

        # Calculate and store CLV
        clv_result = self.calculate_clv(bet_line, closing_line, bet_type)
        self.clv_records.append({
            'game_id': game_id,
            'bet_type': bet_type,
            'clv_result': clv_result,
            'movement': movement
        })

        logger.info(f"CLV for {game_id} {bet_type}: {clv_result.clv_percentage:.2f}% "
                   f"({'BEAT' if clv_result.beat_closing else 'MISSED'} closing)")

        return movement

    def identify_sharp_money(self,
                           line_movement: float,
                           public_betting_percentage: Optional[float] = None) -> bool:
        """
        Identify potential sharp money indicators

        Reverse line movement is a key indicator:
        - Line moves against public betting percentage
        - Significant line movement (>1 point) without news

        Args:
            line_movement: Total line movement
            public_betting_percentage: % of public on one side (optional)

        Returns:
            True if sharp money indicators detected
        """
        # Significant movement threshold
        significant_movement = abs(line_movement) > 1.0

        if public_betting_percentage is not None:
            # Reverse line movement detection
            # If 70% of public on home but line moves toward away
            if public_betting_percentage > 65:
                # Heavy public on one side
                if line_movement < -0.5:
                    # Line moved against public
                    logger.info("Sharp money indicator: Reverse line movement detected")
                    return True
            elif public_betting_percentage < 35:
                # Heavy public on other side
                if line_movement > 0.5:
                    # Line moved against public
                    logger.info("Sharp money indicator: Reverse line movement detected")
                    return True

        # Steam move detection (rapid significant movement)
        if significant_movement:
            logger.info(f"Potential steam move: {line_movement:.1f} point movement")
            return True

        return False

    def analyze_clv_performance(self,
                               period_days: int = 30) -> Dict[str, Union[float, int, Dict]]:
        """
        Analyze CLV performance over recent period

        Research shows:
        - +2% CLV = very good
        - 79.7% of CLV-positive bettors are profitable

        Args:
            period_days: Days to analyze

        Returns:
            Comprehensive CLV analysis
        """
        if not self.clv_records:
            return {
                'avg_clv': 0.0,
                'median_clv': 0.0,
                'positive_clv_rate': 0.0,
                'total_bets': 0,
                'message': 'No CLV data available'
            }

        # Filter recent records if timestamps available
        recent_records = self.clv_records  # Use all for now

        # Extract CLV values
        clv_values = [r['clv_result'].clv_percentage for r in recent_records]
        positive_clv = [r for r in recent_records if r['clv_result'].beat_closing]
        negative_clv = [r for r in recent_records if not r['clv_result'].beat_closing]

        # Calculate metrics
        analysis = {
            'avg_clv': np.mean(clv_values),
            'median_clv': np.median(clv_values),
            'std_clv': np.std(clv_values),
            'positive_clv_rate': len(positive_clv) / len(recent_records) * 100,
            'total_bets': len(recent_records),

            # Performance breakdown
            'positive_clv_count': len(positive_clv),
            'negative_clv_count': len(negative_clv),

            # Distribution percentiles
            'clv_distribution': {
                'p10': np.percentile(clv_values, 10),
                'p25': np.percentile(clv_values, 25),
                'p50': np.percentile(clv_values, 50),
                'p75': np.percentile(clv_values, 75),
                'p90': np.percentile(clv_values, 90)
            },

            # By bet type
            'by_bet_type': self._analyze_by_bet_type(recent_records),

            # Sharp money indicators
            'sharp_moves_detected': sum(1 for r in recent_records
                                      if abs(r['movement'].line_movement) > 1.5)
        }

        # Add interpretation
        if analysis['avg_clv'] > 2.0:
            analysis['interpretation'] = "EXCELLENT: Strong positive CLV indicates profitable edge"
        elif analysis['avg_clv'] > 0:
            analysis['interpretation'] = "GOOD: Positive CLV suggests long-term profitability"
        else:
            analysis['interpretation'] = "WARNING: Negative CLV indicates no edge vs market"

        # Check if meeting profitability threshold
        if analysis['positive_clv_rate'] > 50:
            analysis['profitability_indicator'] = "POSITIVE: >50% CLV beats = likely profitable"
        else:
            analysis['profitability_indicator'] = "NEGATIVE: <50% CLV beats = review strategy"

        return analysis

    def _analyze_by_bet_type(self, records: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze CLV performance by bet type

        Args:
            records: CLV records to analyze

        Returns:
            Breakdown by bet type
        """
        bet_types = {}

        for record in records:
            bet_type = record['bet_type']
            if bet_type not in bet_types:
                bet_types[bet_type] = []
            bet_types[bet_type].append(record['clv_result'].clv_percentage)

        analysis = {}
        for bet_type, clv_values in bet_types.items():
            if clv_values:
                analysis[bet_type] = {
                    'avg_clv': np.mean(clv_values),
                    'positive_rate': sum(1 for v in clv_values if v > 0) / len(clv_values) * 100,
                    'count': len(clv_values)
                }

        return analysis

    def estimate_closing_line(self,
                            current_line: float,
                            hours_until_game: float,
                            bet_type: str = 'spread',
                            historical_movement: Optional[Dict] = None) -> float:
        """
        Estimate where the closing line will be

        Uses historical patterns to predict line movement

        Args:
            current_line: Current line
            hours_until_game: Hours remaining until game start
            bet_type: Type of bet
            historical_movement: Optional historical movement patterns

        Returns:
            Estimated closing line
        """
        # Default movement rates (points per hour)
        default_rates = {
            'spread': 0.05,  # ~0.05 points per hour average movement
            'total': 0.1,     # Totals move more
            'moneyline': 0.02  # Moneyline moves less
        }

        rate = default_rates.get(bet_type, 0.05)

        # Adjust based on time remaining (more movement early)
        if hours_until_game > 48:
            rate *= 1.5  # More movement expected
        elif hours_until_game < 6:
            rate *= 0.5  # Less movement near game time

        # Use historical patterns if available
        if historical_movement:
            avg_movement = historical_movement.get('avg_total_movement', 0)
            if avg_movement != 0:
                rate = avg_movement / 72  # Assume 72-hour window average

        # Estimate movement (could be positive or negative)
        # This is simplified - in production would use ML model
        estimated_movement = rate * hours_until_game * np.random.choice([-1, 1])

        return current_line + estimated_movement

    def generate_clv_report(self) -> str:
        """
        Generate comprehensive CLV report

        Returns:
            Formatted report string
        """
        analysis = self.analyze_clv_performance()

        report = "=" * 70 + "\n"
        report += "CLOSING LINE VALUE (CLV) ANALYSIS REPORT\n"
        report += "=" * 70 + "\n\n"

        report += f"Total Bets Tracked: {analysis['total_bets']}\n"
        report += f"Average CLV: {analysis['avg_clv']:.2f}%\n"
        report += f"Median CLV: {analysis['median_clv']:.2f}%\n"
        report += f"Positive CLV Rate: {analysis['positive_clv_rate']:.1f}%\n\n"

        report += "CLV Distribution:\n"
        report += "-" * 30 + "\n"
        dist = analysis['clv_distribution']
        report += f"  10th percentile: {dist['p10']:.2f}%\n"
        report += f"  25th percentile: {dist['p25']:.2f}%\n"
        report += f"  50th percentile: {dist['p50']:.2f}%\n"
        report += f"  75th percentile: {dist['p75']:.2f}%\n"
        report += f"  90th percentile: {dist['p90']:.2f}%\n\n"

        report += "Performance by Bet Type:\n"
        report += "-" * 30 + "\n"
        for bet_type, metrics in analysis['by_bet_type'].items():
            report += f"  {bet_type.upper()}:\n"
            report += f"    Average CLV: {metrics['avg_clv']:.2f}%\n"
            report += f"    Positive Rate: {metrics['positive_rate']:.1f}%\n"
            report += f"    Total Bets: {metrics['count']}\n"

        report += "\n" + "=" * 70 + "\n"
        report += f"INTERPRETATION: {analysis.get('interpretation', 'N/A')}\n"
        report += f"PROFITABILITY: {analysis.get('profitability_indicator', 'N/A')}\n"
        report += "=" * 70 + "\n"

        report += "\nResearch Note: 79.7% of bettors beating CLV are profitable long-term\n"
        report += "Target: Maintain >50% positive CLV rate and >2% average CLV\n"

        return report


class CLVDatabase:
    """
    Database integration for CLV tracking
    Stores and retrieves CLV data for analysis
    """

    def __init__(self, db_session=None):
        """
        Initialize database connection

        Args:
            db_session: SQLAlchemy session (optional)
        """
        self.db = db_session
        self.cache = {}  # In-memory cache for performance

    def store_clv_record(self,
                        game_id: str,
                        prediction_id: str,
                        clv_result: CLVResult,
                        line_movement: LineMovement) -> bool:
        """
        Store CLV record in database

        Args:
            game_id: Game identifier
            prediction_id: Prediction identifier
            clv_result: CLV calculation result
            line_movement: Line movement details

        Returns:
            Success status
        """
        # For now, store in cache (would integrate with actual DB)
        key = f"{game_id}_{prediction_id}"
        self.cache[key] = {
            'clv_result': clv_result,
            'line_movement': line_movement,
            'timestamp': datetime.now()
        }
        return True

    def get_historical_clv(self,
                          sport: str = None,
                          bet_type: str = None,
                          days_back: int = 30) -> pd.DataFrame:
        """
        Retrieve historical CLV data

        Args:
            sport: Filter by sport (optional)
            bet_type: Filter by bet type (optional)
            days_back: Days of history to retrieve

        Returns:
            DataFrame with CLV history
        """
        # Convert cache to DataFrame for analysis
        if not self.cache:
            return pd.DataFrame()

        records = []
        cutoff = datetime.now() - timedelta(days=days_back)

        for key, data in self.cache.items():
            if data['timestamp'] >= cutoff:
                record = {
                    'key': key,
                    'clv_percentage': data['clv_result'].clv_percentage,
                    'beat_closing': data['clv_result'].beat_closing,
                    'bet_type': data['line_movement'].bet_type,
                    'timestamp': data['timestamp']
                }
                records.append(record)

        df = pd.DataFrame(records)

        # Apply filters
        if bet_type and not df.empty:
            df = df[df['bet_type'] == bet_type]

        return df