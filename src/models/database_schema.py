"""
Database Schema for GameLens.ai Historical Data Storage
Optimized for 3+ years of historical game data and current season stats
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, Text, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional

Base = declarative_base()


class GamesHistory(Base):
    """
    Historical games table - ALL past games since 2010
    Optimized for backtesting and model training
    Storage: ~10GB for 10 years of all sports
    """
    __tablename__ = 'games_history'

    # Primary key
    game_id = Column(String(100), primary_key=True)

    # Game identification
    league = Column(String(20), nullable=False, index=True)  # NBA, NFL, MLB, NHL, etc.
    season = Column(Integer, nullable=False, index=True)     # 2023, 2024, etc.
    game_date = Column(Date, nullable=False, index=True)
    game_datetime = Column(DateTime, nullable=True)

    # Teams
    home_team_id = Column(String(50), nullable=False, index=True)
    away_team_id = Column(String(50), nullable=False, index=True)
    home_team_name = Column(String(100), nullable=False)
    away_team_name = Column(String(100), nullable=False)

    # Scores
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)

    # Game details
    status = Column(String(20), nullable=False)  # final, postponed, cancelled
    venue = Column(String(200), nullable=True)
    attendance = Column(Integer, nullable=True)
    neutral_site = Column(Boolean, default=False)

    # Betting lines (at game time)
    spread = Column(Float, nullable=True)        # Negative = home favored
    total = Column(Float, nullable=True)         # Over/under line
    home_moneyline = Column(Integer, nullable=True)  # American odds
    away_moneyline = Column(Integer, nullable=True)

    # Betting outcomes
    home_covered = Column(Boolean, nullable=True)  # Did home cover spread?
    went_over = Column(Boolean, nullable=True)     # Did game go over total?

    # Additional context
    home_rest_days = Column(Integer, nullable=True)
    away_rest_days = Column(Integer, nullable=True)
    is_playoff = Column(Boolean, default=False)

    # Metadata
    data_source = Column(String(50), nullable=True)  # espn, goalserve
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes for common queries
    __table_args__ = (
        Index('idx_league_date', 'league', 'game_date'),
        Index('idx_home_team_date', 'home_team_id', 'game_date'),
        Index('idx_away_team_date', 'away_team_id', 'game_date'),
        Index('idx_season_league', 'season', 'league'),
    )


class TeamStatsDaily(Base):
    """
    Daily snapshots of team statistics
    Used for tracking performance over time and caching
    Storage: ~1MB per day × 365 days × 3 years = ~1GB
    """
    __tablename__ = 'team_stats_daily'

    # Composite primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(String(50), nullable=False, index=True)
    league = Column(String(20), nullable=False, index=True)
    snapshot_date = Column(Date, nullable=False, index=True)
    season = Column(Integer, nullable=False, index=True)

    # Basic record
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    games_played = Column(Integer, default=0)
    win_pct = Column(Float, nullable=True)

    # Scoring
    points_for = Column(Float, default=0)
    points_against = Column(Float, default=0)
    ppg = Column(Float, nullable=True)          # Points per game
    papg = Column(Float, nullable=True)         # Points against per game

    # Home/Away splits
    home_wins = Column(Integer, default=0)
    home_losses = Column(Integer, default=0)
    away_wins = Column(Integer, default=0)
    away_losses = Column(Integer, default=0)
    home_ppg = Column(Float, nullable=True)
    away_ppg = Column(Float, nullable=True)
    home_papg = Column(Float, nullable=True)
    away_papg = Column(Float, nullable=True)

    # Recent form (last 10 games)
    last_10_wins = Column(Integer, default=0)
    last_10_losses = Column(Integer, default=0)
    recent_ppg = Column(Float, nullable=True)
    recent_papg = Column(Float, nullable=True)

    # Advanced metrics
    elo_rating = Column(Float, nullable=True)
    pythagorean_win_pct = Column(Float, nullable=True)
    strength_of_schedule = Column(Float, nullable=True)

    # Streaks
    current_streak = Column(Integer, default=0)  # Positive = win streak, negative = loss streak

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_team_date', 'team_id', 'snapshot_date'),
        Index('idx_league_season_date', 'league', 'season', 'snapshot_date'),
    )


class OddsHistory(Base):
    """
    Historical betting odds/lines
    Tracks odds movements and closing lines
    Storage: ~5GB for 10 years
    """
    __tablename__ = 'odds_history'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Link to game
    game_id = Column(String(100), ForeignKey('games_history.game_id'), nullable=False, index=True)

    # Odds details
    bookmaker = Column(String(50), nullable=False)
    market_type = Column(String(20), nullable=False)  # h2h, spreads, totals

    # Lines
    spread = Column(Float, nullable=True)
    spread_home_odds = Column(Integer, nullable=True)  # American odds
    spread_away_odds = Column(Integer, nullable=True)

    total = Column(Float, nullable=True)
    over_odds = Column(Integer, nullable=True)
    under_odds = Column(Integer, nullable=True)

    home_moneyline = Column(Integer, nullable=True)
    away_moneyline = Column(Integer, nullable=True)

    # CLV Tracking (NEW)
    line_type = Column(String(20), nullable=True, index=True)  # 'opening', 'bet_placed', 'closing'
    line_moved_from = Column(Float, nullable=True)  # Previous line
    total_line_movement = Column(Float, nullable=True)  # From opening to current
    bet_volume_percentage = Column(Float, nullable=True)  # Public betting % if available
    sharp_money_indicator = Column(Boolean, nullable=True)  # Reverse line movement detected

    # Timing
    odds_datetime = Column(DateTime, nullable=False, index=True)
    is_closing_line = Column(Boolean, default=False)  # Final odds before game

    # Metadata
    data_source = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_game_bookmaker', 'game_id', 'bookmaker'),
        Index('idx_game_closing', 'game_id', 'is_closing_line'),
    )


class TeamRatingsHistory(Base):
    """
    Historical Elo and other rating systems
    Tracks rating changes over time
    Storage: ~100MB for 10 years
    """
    __tablename__ = 'team_ratings_history'

    id = Column(Integer, primary_key=True, autoincrement=True)

    team_id = Column(String(50), nullable=False, index=True)
    league = Column(String(20), nullable=False, index=True)
    rating_date = Column(Date, nullable=False, index=True)
    season = Column(Integer, nullable=False, index=True)

    # Elo ratings
    elo_rating = Column(Float, nullable=False)
    elo_offense = Column(Float, nullable=True)
    elo_defense = Column(Float, nullable=True)

    # Pythagorean
    pythagorean_win_pct = Column(Float, nullable=True)
    expected_wins = Column(Float, nullable=True)
    actual_wins = Column(Integer, nullable=True)
    luck_factor = Column(Float, nullable=True)  # Difference between actual and expected

    # Context
    games_played = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_team_date_rating', 'team_id', 'rating_date'),
        Index('idx_league_season', 'league', 'season'),
    )


class PredictionsLog(Base):
    """
    Log of all predictions made by the system
    Used for tracking model performance and Sharpe ratios
    Storage: ~500MB per year
    """
    __tablename__ = 'predictions_log'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Link to game
    game_id = Column(String(100), ForeignKey('games_history.game_id'), nullable=False, index=True)

    # Prediction details
    model_name = Column(String(50), nullable=False, index=True)  # monte_carlo, elo, ensemble, etc.

    # Predictions
    predicted_home_win_pct = Column(Float, nullable=False)
    predicted_spread = Column(Float, nullable=True)
    predicted_total = Column(Float, nullable=True)
    predicted_home_cover_pct = Column(Float, nullable=True)
    predicted_over_pct = Column(Float, nullable=True)

    # Confidence
    confidence = Column(Float, nullable=True)
    model_agreement = Column(Float, nullable=True)  # For ensemble only

    # Recommendation
    recommended_bet = Column(String(100), nullable=True)  # "HOME -5.5", "OVER 218", etc.
    kelly_stake = Column(Float, nullable=True)  # Percentage of bankroll

    # Market context
    market_spread = Column(Float, nullable=True)
    market_total = Column(Float, nullable=True)
    edge = Column(Float, nullable=True)  # Our prediction vs market

    # Actual outcomes (filled after game)
    actual_home_win = Column(Boolean, nullable=True)
    actual_home_score = Column(Integer, nullable=True)
    actual_away_score = Column(Integer, nullable=True)
    actual_home_covered = Column(Boolean, nullable=True)
    actual_went_over = Column(Boolean, nullable=True)

    # Performance metrics (calculated after game)
    prediction_correct = Column(Boolean, nullable=True)
    brier_score = Column(Float, nullable=True)
    profit_loss = Column(Float, nullable=True)  # If bet was placed

    # CLV Tracking (NEW)
    opening_line = Column(Float, nullable=True)
    bet_line = Column(Float, nullable=True)  # Line when bet was placed
    closing_line = Column(Float, nullable=True)
    clv_percentage = Column(Float, nullable=True)  # (bet_odds - close_odds) / close_odds
    clv_points = Column(Float, nullable=True)  # Absolute difference
    beat_closing_line = Column(Boolean, nullable=True)  # True if CLV > 0

    # Calibration Tracking (NEW)
    uncalibrated_probability = Column(Float, nullable=True)
    calibrated_probability = Column(Float, nullable=True)
    calibration_method = Column(String(20), nullable=True)  # 'platt' or 'isotonic'
    calibration_adjustment = Column(Float, nullable=True)  # difference

    # Timing
    prediction_datetime = Column(DateTime, default=datetime.utcnow, index=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_model_date', 'model_name', 'prediction_datetime'),
        Index('idx_game_model', 'game_id', 'model_name'),
    )


class ModelPerformance(Base):
    """
    Aggregated model performance metrics
    Updated daily for tracking model accuracy over time
    """
    __tablename__ = 'model_performance'

    id = Column(Integer, primary_key=True, autoincrement=True)

    model_name = Column(String(50), nullable=False, index=True)
    league = Column(String(20), nullable=False, index=True)

    # Time period
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)

    # Performance metrics
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    accuracy = Column(Float, nullable=True)

    avg_brier_score = Column(Float, nullable=True)
    avg_confidence = Column(Float, nullable=True)

    # Betting performance
    total_bets = Column(Integer, default=0)
    winning_bets = Column(Integer, default=0)
    total_profit_loss = Column(Float, default=0.0)
    roi = Column(Float, nullable=True)  # Return on investment
    sharpe_ratio = Column(Float, nullable=True)

    # By bet type
    win_accuracy = Column(Float, nullable=True)
    ats_accuracy = Column(Float, nullable=True)
    ou_accuracy = Column(Float, nullable=True)

    # Updated
    calculated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_model_league_period', 'model_name', 'league', 'period_end'),
    )


class CLVAnalytics(Base):
    """
    Aggregate CLV (Closing Line Value) performance by model, sport, bet type
    Critical for long-term profitability tracking
    Research: 79.7% of CLV-positive bettors are profitable
    """
    __tablename__ = 'clv_analytics'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identifiers
    model_name = Column(String(50), nullable=False, index=True)
    sport = Column(String(20), nullable=False, index=True)
    bet_type = Column(String(20), nullable=False)  # 'spread', 'moneyline', 'total'

    # Time period
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)

    # CLV metrics
    avg_clv_percentage = Column(Float, nullable=False)
    median_clv_percentage = Column(Float, nullable=True)
    positive_clv_rate = Column(Float, nullable=False)  # % of bets with CLV > 0
    total_bets = Column(Integer, nullable=False)

    # Performance correlation
    roi_with_positive_clv = Column(Float, nullable=True)
    roi_with_negative_clv = Column(Float, nullable=True)
    win_rate_positive_clv = Column(Float, nullable=True)
    win_rate_negative_clv = Column(Float, nullable=True)

    # Sharp money indicators
    sharp_moves_detected = Column(Integer, default=0)
    reverse_line_movements = Column(Integer, default=0)

    # Metadata
    calculated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_clv_model_sport', 'model_name', 'sport', 'period_end'),
        Index('idx_clv_period', 'period_start', 'period_end'),
    )


class CalibrationMetrics(Base):
    """
    Track calibration performance over time
    Target: ECE < 0.05 for production deployment
    Calibration adds +34.69% ROI per research
    """
    __tablename__ = 'calibration_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identifiers
    model_name = Column(String(50), nullable=False, index=True)
    sport = Column(String(20), nullable=True, index=True)
    bet_type = Column(String(20), nullable=True)  # 'win', 'ats', 'ou'

    # Time period
    date = Column(Date, nullable=False, index=True)

    # Calibration metrics
    ece = Column(Float, nullable=False)  # Expected Calibration Error (target < 0.05)
    mce = Column(Float, nullable=True)   # Maximum Calibration Error
    brier_score_uncalibrated = Column(Float, nullable=True)
    brier_score_calibrated = Column(Float, nullable=True)
    calibration_method = Column(String(20), nullable=True)  # 'isotonic' or 'platt'

    # Sample statistics
    n_predictions = Column(Integer, nullable=False)
    n_bins = Column(Integer, default=10)

    # Per-bin calibration (stored as JSON strings)
    bin_accuracies = Column(Text)  # JSON array of actual accuracies per bin
    bin_confidences = Column(Text)  # JSON array of predicted confidences per bin
    bin_counts = Column(Text)  # JSON array of sample counts per bin

    # Performance improvement
    ece_improvement_pct = Column(Float, nullable=True)
    brier_improvement_pct = Column(Float, nullable=True)
    meets_production_threshold = Column(Boolean, default=False)  # ECE < 0.05

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_calibration_model_date', 'model_name', 'date'),
        Index('idx_calibration_threshold', 'meets_production_threshold'),
    )


# Helper function to create all tables
def create_all_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(engine)


# Helper function to get current season stats for a team (fast query)
def get_current_team_stats_query(team_id: str, league: str):
    """
    SQL query to get most recent team stats
    Use this as a template for fast queries
    """
    return f"""
    SELECT * FROM team_stats_daily
    WHERE team_id = '{team_id}'
    AND league = '{league}'
    ORDER BY snapshot_date DESC
    LIMIT 1
    """


# Helper function to get team history for training
def get_team_game_history_query(team_id: str, league: str, days_back: int = 365):
    """
    SQL query to get team's game history
    Use for calculating recent form, trends, etc.
    """
    return f"""
    SELECT * FROM games_history
    WHERE (home_team_id = '{team_id}' OR away_team_id = '{team_id}')
    AND league = '{league}'
    AND game_date >= CURRENT_DATE - INTERVAL '{days_back} days'
    AND status = 'final'
    ORDER BY game_date DESC
    """


if __name__ == "__main__":
    # Example: Create tables in PostgreSQL
    from sqlalchemy import create_engine
    import os

    # Get database URL from environment or use default
    DATABASE_URL = os.getenv(
        'DATABASE_URL',
        'postgresql://user:password@localhost:5432/gamelens'
    )

    print("Creating GameLens.ai database schema...")
    print(f"Database: {DATABASE_URL}")

    engine = create_engine(DATABASE_URL)

    # Create all tables
    create_all_tables(engine)

    print("✅ All tables created successfully!")
    print()
    print("Tables created:")
    print("  • games_history - Historical games since 2010")
    print("  • team_stats_daily - Daily team statistics snapshots")
    print("  • odds_history - Historical betting lines")
    print("  • team_ratings_history - Elo/Pythagorean ratings over time")
    print("  • predictions_log - All system predictions")
    print("  • model_performance - Aggregated model metrics")
