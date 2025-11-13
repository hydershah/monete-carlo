"""
SQLAlchemy models for sports prediction system.
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Text, JSON, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from .database import Base


class Team(Base):
    """Team model for storing team information."""

    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True)  # ESPN/API team ID
    sport = Column(String, index=True)  # 'nba', 'nfl', etc.
    league = Column(String)  # Full league name
    name = Column(String)
    abbreviation = Column(String)
    display_name = Column(String)
    location = Column(String, nullable=True)
    logo_url = Column(String, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    home_games = relationship("Game", back_populates="home_team", foreign_keys="Game.home_team_id")
    away_games = relationship("Game", back_populates="away_team", foreign_keys="Game.away_team_id")
    elo_ratings = relationship("EloRating", back_populates="team")

    __table_args__ = (
        Index('ix_teams_sport_league', 'sport', 'league'),
    )


class Game(Base):
    """Game model for storing game information and results."""

    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True)  # ESPN game ID
    sport = Column(String, index=True)
    league = Column(String)

    # Teams
    home_team_id = Column(Integer, ForeignKey("teams.id"))
    away_team_id = Column(Integer, ForeignKey("teams.id"))

    # Game details
    game_date = Column(DateTime, index=True)
    status = Column(String)  # 'scheduled', 'in_progress', 'completed', 'postponed'
    venue = Column(String, nullable=True)
    attendance = Column(Integer, nullable=True)

    # Scores
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)

    # Pre-game team records
    home_team_record = Column(String, nullable=True)
    away_team_record = Column(String, nullable=True)

    # Odds and betting information
    odds_data = Column(JSON, nullable=True)  # Store odds from TheOddsAPI

    # Additional metadata
    metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    home_team = relationship("Team", back_populates="home_games", foreign_keys=[home_team_id])
    away_team = relationship("Team", back_populates="away_games", foreign_keys=[away_team_id])
    predictions = relationship("Prediction", back_populates="game")

    __table_args__ = (
        Index('ix_games_date_sport', 'game_date', 'sport'),
        Index('ix_games_teams', 'home_team_id', 'away_team_id'),
    )


class EloRating(Base):
    """Elo rating history for teams."""

    __tablename__ = "elo_ratings"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), index=True)
    sport = Column(String, index=True)

    # Rating
    rating = Column(Float)
    rating_date = Column(DateTime, index=True)

    # Context
    season = Column(String, nullable=True)  # e.g., '2024-2025'
    games_played = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    team = relationship("Team", back_populates="elo_ratings")

    __table_args__ = (
        Index('ix_elo_team_date', 'team_id', 'rating_date'),
    )


class Prediction(Base):
    """Prediction model for storing model predictions."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), index=True)

    # Prediction metadata
    model_version = Column(String)  # 'v1.0', 'monte_carlo', etc.
    prediction_date = Column(DateTime, server_default=func.now(), index=True)

    # Predictions
    home_win_probability = Column(Float)  # 0.0 to 1.0
    away_win_probability = Column(Float)  # 0.0 to 1.0
    draw_probability = Column(Float, nullable=True)  # For sports with draws

    # Score predictions
    predicted_home_score = Column(Float, nullable=True)
    predicted_away_score = Column(Float, nullable=True)

    # Spread predictions
    predicted_spread = Column(Float, nullable=True)  # Positive = home favored
    predicted_total = Column(Float, nullable=True)  # Over/under total

    # Model-specific data
    monte_carlo_iterations = Column(Integer, nullable=True)
    elo_home = Column(Float, nullable=True)
    elo_away = Column(Float, nullable=True)

    # GPT analysis
    gpt_analysis = Column(Text, nullable=True)
    gpt_confidence = Column(Float, nullable=True)  # 0.0 to 1.0
    gpt_adjustment = Column(Float, nullable=True)  # Adjustment to base probability

    # Final combined prediction
    final_home_win_prob = Column(Float)  # After combining quant + GPT
    final_away_win_prob = Column(Float)

    # Confidence and metrics
    confidence_level = Column(Float)  # 0.0 to 1.0
    edge_detected = Column(Float, nullable=True)  # vs betting odds
    recommended_bet = Column(String, nullable=True)  # 'home', 'away', 'pass', etc.

    # Validation (filled in after game completes)
    actual_outcome = Column(String, nullable=True)  # 'home_win', 'away_win', 'draw'
    prediction_correct = Column(Boolean, nullable=True)
    brier_score = Column(Float, nullable=True)  # Calibration metric

    # Additional data
    metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    game = relationship("Game", back_populates="predictions")

    __table_args__ = (
        Index('ix_predictions_game_model', 'game_id', 'model_version'),
        Index('ix_predictions_date', 'prediction_date'),
    )


class TeamStats(Base):
    """Time-series team statistics."""

    __tablename__ = "team_stats"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), index=True)
    stat_date = Column(DateTime, index=True)
    season = Column(String)

    # Offensive stats
    points_per_game = Column(Float, nullable=True)
    offensive_rating = Column(Float, nullable=True)
    field_goal_percentage = Column(Float, nullable=True)
    three_point_percentage = Column(Float, nullable=True)

    # Defensive stats
    points_allowed_per_game = Column(Float, nullable=True)
    defensive_rating = Column(Float, nullable=True)
    opponent_field_goal_pct = Column(Float, nullable=True)

    # General stats
    wins = Column(Integer, nullable=True)
    losses = Column(Integer, nullable=True)
    win_percentage = Column(Float, nullable=True)
    home_record = Column(String, nullable=True)
    away_record = Column(String, nullable=True)

    # Advanced metrics (sport-specific)
    advanced_stats = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('ix_team_stats_team_date', 'team_id', 'stat_date'),
        UniqueConstraint('team_id', 'stat_date', name='uq_team_stats_team_date'),
    )


class PerformanceMetrics(Base):
    """Track model performance over time."""

    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True)

    # Metadata
    model_version = Column(String, index=True)
    sport = Column(String, index=True)
    evaluation_date = Column(DateTime, index=True)
    date_range_start = Column(DateTime)
    date_range_end = Column(DateTime)

    # Performance metrics
    total_predictions = Column(Integer)
    correct_predictions = Column(Integer)
    accuracy = Column(Float)  # Percentage correct

    # Calibration metrics
    average_brier_score = Column(Float)
    calibration_error = Column(Float)

    # Betting metrics
    roi = Column(Float, nullable=True)  # Return on investment
    profit_loss = Column(Float, nullable=True)
    total_bet_amount = Column(Float, nullable=True)
    positive_clv_count = Column(Integer, nullable=True)  # Closing line value
    average_clv = Column(Float, nullable=True)

    # Risk metrics
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)

    # Additional metrics
    metrics_data = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index('ix_perf_metrics_model_sport', 'model_version', 'sport'),
        Index('ix_perf_metrics_date', 'evaluation_date'),
    )


class APIUsage(Base):
    """Track API usage and costs."""

    __tablename__ = "api_usage"

    id = Column(Integer, primary_key=True, index=True)

    # API details
    api_name = Column(String, index=True)  # 'espn', 'theodds', 'openai'
    endpoint = Column(String, nullable=True)
    request_type = Column(String, nullable=True)  # 'GET', 'POST', etc.

    # Usage
    request_date = Column(DateTime, server_default=func.now(), index=True)
    response_status = Column(Integer, nullable=True)
    response_time_ms = Column(Integer, nullable=True)

    # Cost tracking
    tokens_used = Column(Integer, nullable=True)  # For LLM APIs
    estimated_cost = Column(Float, nullable=True)

    # Additional data
    metadata = Column(JSON, nullable=True)

    __table_args__ = (
        Index('ix_api_usage_name_date', 'api_name', 'request_date'),
    )
