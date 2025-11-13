"""
Test script for the Sports Prediction API.
Demonstrates how to call the API endpoints.
"""

import requests
import json

# API base URL
API_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint."""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)

    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_single_prediction():
    """Test single game prediction."""
    print("\n" + "="*60)
    print("Testing Single Game Prediction")
    print("="*60)

    game_data = {
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
        "recent_form": "Celtics won last 4 games, Lakers 2-2 in last 4",
        "use_gpt": True,
        "monte_carlo_iterations": 10000
    }

    print("\nRequest:")
    print(json.dumps(game_data, indent=2))

    response = requests.post(
        f"{API_URL}/api/v1/predict",
        json=game_data
    )

    print(f"\nStatus: {response.status_code}")
    print("\nResponse:")
    result = response.json()
    print(json.dumps(result, indent=2))

    print("\n" + "-"*60)
    print("PREDICTION SUMMARY")
    print("-"*60)
    print(f"Game: {result['away_team']} @ {result['home_team']}")
    print(f"Home Win: {result['home_win_probability']:.1%}")
    print(f"Away Win: {result['away_win_probability']:.1%}")
    print(f"Expected Score: {result['predicted_home_score']:.1f} - {result['predicted_away_score']:.1f}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Recommendation: {result['recommendation']}")
    if result.get('gpt_analysis'):
        print(f"\nGPT Analysis:")
        print(f"  {result['gpt_analysis'][:200]}...")
    print(f"\nProcessing Time: {result['processing_time_ms']:.0f}ms")


def test_batch_prediction():
    """Test batch prediction."""
    print("\n" + "="*60)
    print("Testing Batch Prediction")
    print("="*60)

    batch_data = {
        "games": [
            {
                "home_team": "Lakers",
                "away_team": "Celtics",
                "sport": "nba",
                "home_ppg": 112.5,
                "away_ppg": 115.2,
                "use_gpt": False  # Disable GPT for faster batch processing
            },
            {
                "home_team": "Warriors",
                "away_team": "Nets",
                "sport": "nba",
                "home_ppg": 118.3,
                "away_ppg": 110.1,
                "use_gpt": False
            },
            {
                "home_team": "Heat",
                "away_team": "Bucks",
                "sport": "nba",
                "home_ppg": 108.7,
                "away_ppg": 116.4,
                "use_gpt": False
            }
        ]
    }

    response = requests.post(
        f"{API_URL}/api/v1/predict/batch",
        json=batch_data
    )

    print(f"Status: {response.status_code}")
    result = response.json()

    print(f"\nProcessed {result['successful_predictions']}/{result['total_games']} games")
    print(f"Total time: {result['total_processing_time_ms']:.0f}ms")

    print("\nPredictions:")
    for pred in result['predictions']:
        print(f"\n  {pred['away_team']} @ {pred['home_team']}")
        print(f"    Prediction: {pred['home_win_probability']:.1%} home")
        print(f"    Recommendation: {pred['recommendation']}")


def test_gpt_only_analysis():
    """Test GPT-only analysis (no Monte Carlo)."""
    print("\n" + "="*60)
    print("Testing GPT-Only Analysis")
    print("="*60)

    game_data = {
        "home_team": "Dallas Cowboys",
        "away_team": "Philadelphia Eagles",
        "sport": "nfl",
        "home_record": "6-3",
        "away_record": "8-1",
        "injuries": "Cowboys: Dak Prescott (out - thumb)",
        "recent_form": "Eagles on 6-game win streak, Cowboys lost 2 of last 3"
    }

    response = requests.post(
        f"{API_URL}/api/v1/analyze",
        json=game_data
    )

    print(f"Status: {response.status_code}")
    result = response.json()

    print(f"\n{result['away_team']} @ {result['home_team']}")
    print(f"Prediction: {result['home_win_probability']:.1%} home")
    print(f"Recommendation: {result['recommendation']}")
    if result.get('gpt_analysis'):
        print(f"\nGPT Analysis: {result['gpt_analysis']}")


def test_chat():
    """Test chat functionality."""
    print("\n" + "="*60)
    print("Testing Chat Interface")
    print("="*60)

    # First get a prediction
    game_data = {
        "home_team": "Lakers",
        "away_team": "Celtics",
        "sport": "nba",
        "home_ppg": 112.5,
        "away_ppg": 115.2,
        "use_gpt": True
    }

    pred_response = requests.post(f"{API_URL}/api/v1/predict", json=game_data)
    prediction = pred_response.json()

    # Format context
    context = f"""
    Game: {prediction['away_team']} @ {prediction['home_team']}
    Prediction: {prediction['home_win_probability']:.1%} home win
    Recommendation: {prediction['recommendation']}
    """

    print("\nPrediction context loaded.")
    print("\nAsking AI: 'What factors favor the Celtics?'")

    # Ask a question
    chat_response = requests.post(
        f"{API_URL}/api/v1/chat",
        json={
            "question": "What factors favor the Celtics in this matchup?",
            "game_context": context
        }
    )

    chat_result = chat_response.json()

    print(f"\nAI Answer:")
    print(f"  {chat_result['answer']}")
    print(f"\nTokens used: {chat_result.get('tokens_used', 'N/A')}")

    # Get suggested questions
    print("\n" + "-"*60)
    print("Getting suggested questions...")

    suggest_response = requests.post(
        f"{API_URL}/api/v1/suggest-questions",
        json={"prediction": prediction}
    )

    suggestions = suggest_response.json()

    print("\nSuggested questions:")
    for i, q in enumerate(suggestions['suggestions'], 1):
        print(f"  {i}. {q}")


def test_supported_sports():
    """Test supported sports endpoint."""
    print("\n" + "="*60)
    print("Testing Supported Sports")
    print("="*60)

    response = requests.get(f"{API_URL}/api/v1/sports")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  SPORTS PREDICTION API - Test Suite")
    print("="*60)

    try:
        # Test health check
        test_health()

        # Test supported sports
        test_supported_sports()

        # Test single prediction
        test_single_prediction()

        # Test batch prediction
        # test_batch_prediction()

        # Test GPT-only analysis
        # test_gpt_only_analysis()

        # Test chat interface
        test_chat()

        print("\n" + "="*60)
        print("  All tests completed!")
        print("="*60 + "\n")

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running:")
        print("  python -m src.api.main")
        print("  or")
        print("  uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
