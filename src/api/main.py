"""
FastAPI application for sports predictions API.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import time
from datetime import datetime
from loguru import logger
import os

from .schemas import (
    GameDataInput,
    PredictionResponse,
    BatchGameInput,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    ChatRequest,
    ChatResponse,
    ExplainRequest,
    SuggestQuestionsRequest
)
from .predictor_service import prediction_service
from .chat_service import chat_service

# Initialize FastAPI app
app = FastAPI(
    title="Sports Prediction API",
    description="AI-powered sports prediction system using Monte Carlo simulations and GPT analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Sports Prediction API",
        "version": "1.0.0",
        "description": "Predict sports game outcomes using AI and Monte Carlo simulations",
        "endpoints": {
            "predict": "POST /api/v1/predict",
            "batch_predict": "POST /api/v1/predict/batch",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns system status and configuration.
    """
    # Check if API keys are configured
    openai_configured = bool(os.getenv("OPENAI_API_KEY"))
    theodds_configured = bool(os.getenv("THEODDS_API_KEY"))

    # Database checks would go here
    db_connected = True  # TODO: Actual DB check
    redis_connected = True  # TODO: Actual Redis check

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        database_connected=db_connected,
        redis_connected=redis_connected,
        api_keys_configured={
            "openai": openai_configured,
            "theodds": theodds_configured
        }
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_game(game_data: GameDataInput):
    """
    Predict the outcome of a single game.

    This endpoint accepts game information and returns a prediction using:
    - Monte Carlo simulation (10,000+ iterations)
    - GPT-4o Mini qualitative analysis
    - Statistical models (Elo, Poisson)

    **Example Request:**
    ```json
    {
      "home_team": "Los Angeles Lakers",
      "away_team": "Boston Celtics",
      "sport": "nba",
      "home_ppg": 112.5,
      "away_ppg": 115.2,
      "home_defensive_rating": 108.3,
      "away_defensive_rating": 106.1,
      "injuries": "Lakers: Anthony Davis (questionable)",
      "use_gpt": true,
      "monte_carlo_iterations": 10000
    }
    ```

    **Response includes:**
    - Win probabilities for each team
    - Expected final score
    - Confidence level
    - Betting recommendation
    - GPT analysis and insights
    """
    try:
        logger.info(
            f"API request: {game_data.away_team} @ {game_data.home_team} "
            f"({game_data.sport})"
        )

        # Make prediction
        prediction = prediction_service.predict_game(game_data)

        return prediction

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(batch_input: BatchGameInput):
    """
    Predict multiple games in batch.

    Accepts a list of games and returns predictions for all of them.
    Can process games in parallel for better performance.

    **Example Request:**
    ```json
    {
      "games": [
        {
          "home_team": "Lakers",
          "away_team": "Celtics",
          "sport": "nba",
          "home_ppg": 112.5,
          "away_ppg": 115.2
        },
        {
          "home_team": "Warriors",
          "away_team": "Nets",
          "sport": "nba",
          "home_ppg": 118.3,
          "away_ppg": 110.1
        }
      ],
      "parallel_processing": true
    }
    ```
    """
    start_time = time.time()

    predictions = []
    failed = 0

    for game in batch_input.games:
        try:
            prediction = prediction_service.predict_game(game)
            predictions.append(prediction)
        except Exception as e:
            logger.error(f"Failed to predict {game.away_team} @ {game.home_team}: {e}")
            failed += 1
            continue

    processing_time = (time.time() - start_time) * 1000  # ms

    return BatchPredictionResponse(
        predictions=predictions,
        total_games=len(batch_input.games),
        successful_predictions=len(predictions),
        failed_predictions=failed,
        total_processing_time_ms=processing_time
    )


@app.post("/api/v1/analyze", tags=["Analysis"])
async def analyze_game_gpt_only(game_data: GameDataInput):
    """
    Get GPT analysis only (no Monte Carlo simulation).

    Faster and cheaper - uses only GPT to analyze the matchup.
    Good for quick insights without full statistical modeling.

    **Use this when:**
    - You want quick qualitative analysis
    - Cost optimization is important
    - You don't need Monte Carlo simulations
    """
    try:
        # Force GPT only
        game_data.use_gpt = True
        game_data.use_monte_carlo = False

        prediction = prediction_service.predict_game(game_data)

        return prediction

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/api/v1/sports", tags=["Information"])
async def get_supported_sports():
    """
    Get list of supported sports.

    Returns all sports that the system can analyze.
    """
    return {
        "supported_sports": [
            {
                "id": "nba",
                "name": "NBA Basketball",
                "league_avg_score": 112.0,
                "models": ["monte_carlo", "elo", "gpt"]
            },
            {
                "id": "nfl",
                "name": "NFL Football",
                "league_avg_score": 23.0,
                "models": ["monte_carlo", "elo", "gpt"]
            },
            {
                "id": "mlb",
                "name": "MLB Baseball",
                "league_avg_score": 4.5,
                "models": ["monte_carlo", "elo", "gpt", "poisson"]
            },
            {
                "id": "nhl",
                "name": "NHL Hockey",
                "league_avg_score": 3.0,
                "models": ["monte_carlo", "elo", "gpt", "poisson"]
            },
            {
                "id": "soccer",
                "name": "Soccer/Football",
                "league_avg_score": 2.7,
                "models": ["monte_carlo", "elo", "gpt", "poisson"]
            }
        ]
    }


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_about_game(chat_request: ChatRequest):
    """
    Chat interface for asking questions about games and predictions.

    This endpoint allows users to ask follow-up questions about predictions,
    get explanations, and have interactive conversations about games.

    **Example Request:**
    ```json
    {
      "question": "What factors favor the Celtics?",
      "game_context": "Lakers vs Celtics - 55% away win probability"
    }
    ```

    **Use Cases:**
    - "Why is Team A favored?"
    - "What could change the outcome?"
    - "Explain the Monte Carlo simulation"
    - "Should I bet on this game?"
    - "What are the key matchups?"
    """
    try:
        result = chat_service.chat(
            user_question=chat_request.question,
            game_context=chat_request.game_context,
            conversation_history=chat_request.conversation_history
        )

        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


@app.post("/api/v1/explain", response_model=ChatResponse, tags=["Chat"])
async def explain_prediction(request: ExplainRequest):
    """
    Explain a specific aspect of a prediction.

    **Aspects you can explain:**
    - `general`: Overall explanation
    - `probability`: How probabilities were calculated
    - `gpt`: GPT analysis factors
    - `monte_carlo`: Monte Carlo simulation
    - `confidence`: Confidence level meaning
    - `recommendation`: Betting recommendation reasoning

    **Example Request:**
    ```json
    {
      "prediction": { ...prediction object... },
      "aspect": "probability"
    }
    ```
    """
    try:
        explanation = chat_service.explain_prediction(
            prediction_data=request.prediction.dict(),
            aspect=request.aspect
        )

        return ChatResponse(
            question=f"Explain {request.aspect}",
            answer=explanation,
            model=chat_service.model
        )

    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {str(e)}"
        )


@app.post("/api/v1/suggest-questions", tags=["Chat"])
async def suggest_questions(request: SuggestQuestionsRequest):
    """
    Get suggested questions users might want to ask about a prediction.

    Returns a list of relevant questions based on the prediction results.

    **Example Response:**
    ```json
    {
      "suggestions": [
        "Why is the home team favored?",
        "What factors most influenced this prediction?",
        "How confident should I be in this prediction?"
      ]
    }
    ```
    """
    try:
        suggestions = chat_service.suggest_questions(
            prediction_data=request.prediction.dict()
        )

        return {
            "suggestions": suggestions
        }

    except Exception as e:
        logger.error(f"Suggest questions failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate suggestions: {str(e)}"
        )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Starting Sports Prediction API v1.0.0")
    logger.info(f"OpenAI configured: {bool(os.getenv('OPENAI_API_KEY'))}")
    logger.info("API ready to accept requests")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down Sports Prediction API")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
