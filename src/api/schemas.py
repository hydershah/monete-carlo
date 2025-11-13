"""
Pydantic schemas for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class GameDataInput(BaseModel):
    """Input schema for game prediction request."""

    # Required fields
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    sport: str = Field(..., description="Sport (nba, nfl, mlb, nhl, soccer)")

    # Optional context
    game_date: Optional[str] = Field(None, description="Game date (YYYY-MM-DD)")
    home_record: Optional[str] = Field(None, description="Home team record (e.g., '10-5')")
    away_record: Optional[str] = Field(None, description="Away team record (e.g., '8-7')")

    # Team statistics
    home_ppg: Optional[float] = Field(None, description="Home team points per game")
    away_ppg: Optional[float] = Field(None, description="Away team points per game")
    home_offensive_rating: Optional[float] = Field(None, description="Home offensive rating")
    away_offensive_rating: Optional[float] = Field(None, description="Away offensive rating")
    home_defensive_rating: Optional[float] = Field(None, description="Home defensive rating")
    away_defensive_rating: Optional[float] = Field(None, description="Away defensive rating")

    # Contextual information
    injuries: Optional[str] = Field(None, description="Injury report")
    recent_form: Optional[str] = Field(None, description="Recent form/momentum")
    head_to_head: Optional[str] = Field(None, description="Head-to-head history")
    betting_odds: Optional[Dict[str, float]] = Field(None, description="Betting odds if available")

    # Analysis options
    use_gpt: bool = Field(True, description="Use GPT analysis")
    use_monte_carlo: bool = Field(True, description="Use Monte Carlo simulation")
    monte_carlo_iterations: int = Field(10000, description="Number of MC iterations")

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    class Config:
        json_schema_extra = {
            "example": {
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "sport": "nba",
                "game_date": "2024-11-13",
                "home_record": "10-5",
                "away_record": "12-3",
                "home_ppg": 112.5,
                "away_ppg": 115.2,
                "home_defensive_rating": 108.3,
                "away_defensive_rating": 106.1,
                "injuries": "Lakers: Anthony Davis (questionable - ankle)",
                "recent_form": "Celtics won last 4 games, Lakers split last 4",
                "use_gpt": True,
                "monte_carlo_iterations": 10000
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction."""

    # Game info
    home_team: str
    away_team: str
    sport: str
    game_date: Optional[str] = None

    # Predictions
    home_win_probability: float = Field(..., description="Home team win probability (0-1)")
    away_win_probability: float = Field(..., description="Away team win probability (0-1)")
    draw_probability: Optional[float] = Field(None, description="Draw probability if applicable")

    # Expected scores
    predicted_home_score: Optional[float] = None
    predicted_away_score: Optional[float] = None

    # Confidence and recommendation
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    recommendation: str = Field(..., description="Betting recommendation")

    # Model details
    models_used: List[str] = Field(..., description="Models used in prediction")
    gpt_analysis: Optional[str] = None
    gpt_adjustment: Optional[float] = None

    # Additional data
    most_likely_score: Optional[str] = None
    spread_analysis: Optional[Dict[str, float]] = None
    total_analysis: Optional[Dict[str, float]] = None

    # Metadata
    prediction_timestamp: str
    processing_time_ms: Optional[float] = None


class BatchGameInput(BaseModel):
    """Input schema for batch prediction."""

    games: List[GameDataInput] = Field(..., description="List of games to predict")
    parallel_processing: bool = Field(True, description="Process games in parallel")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictionResponse]
    total_games: int
    successful_predictions: int
    failed_predictions: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str
    database_connected: bool
    redis_connected: bool
    api_keys_configured: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: Optional[str] = None
    timestamp: str


class ChatRequest(BaseModel):
    """Chat request schema."""

    question: str = Field(..., description="User's question about the game")
    game_context: Optional[str] = Field(None, description="Previous prediction/analysis context")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Previous messages in conversation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What factors favor the Celtics in this matchup?",
                "game_context": "Lakers vs Celtics prediction showing 55% away win probability",
                "conversation_history": [
                    {"role": "user", "content": "How was this calculated?"},
                    {"role": "assistant", "content": "The prediction combines..."}
                ]
            }
        }


class ChatResponse(BaseModel):
    """Chat response schema."""

    question: str
    answer: str
    tokens_used: Optional[int] = None
    model: Optional[str] = None
    suggested_questions: Optional[List[str]] = None


class ExplainRequest(BaseModel):
    """Request to explain a prediction."""

    prediction: PredictionResponse
    aspect: str = Field(
        "general",
        description="What to explain: general, probability, gpt, monte_carlo, confidence, recommendation"
    )


class SuggestQuestionsRequest(BaseModel):
    """Request for suggested questions."""

    prediction: PredictionResponse
