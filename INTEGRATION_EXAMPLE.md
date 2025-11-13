# Website Integration Example

This guide shows how to integrate the Sports Prediction API into your website for **on-demand game analysis with chat interface**.

## 🎯 Your Workflow

```
Your Website → User clicks "Analyze" → Fetch game data → Send to AI API → Show prediction + Chat
```

---

## 📋 Complete Integration Example

### 1. Backend Integration (Your Server)

```python
# your_backend.py
import requests
from fastapi import FastAPI, HTTPException

app = FastAPI()

# AI API configuration
AI_API_URL = "http://localhost:8000"  # or your deployed URL

@app.post("/api/analyze-game/{game_id}")
async def analyze_game(game_id: str):
    """
    Called when user clicks "Analyze" button on your website.

    Flow:
    1. Fetch game data from your DB or ESPN
    2. Send to AI API for analysis
    3. Return prediction to frontend
    """

    # Step 1: Get game data (from your database or ESPN API)
    game = fetch_game_from_database(game_id)
    # Returns: {
    #   home_team: "Lakers",
    #   away_team: "Celtics",
    #   home_ppg: 112.5,
    #   away_ppg: 115.2,
    #   injuries: "...",
    #   ...
    # }

    # Step 2: Send to AI API for analysis
    prediction_response = requests.post(
        f"{AI_API_URL}/api/v1/predict",
        json={
            "home_team": game['home_team'],
            "away_team": game['away_team'],
            "sport": game['sport'],
            "home_ppg": game['home_ppg'],
            "away_ppg": game['away_ppg'],
            "home_defensive_rating": game['home_def_rating'],
            "away_defensive_rating": game['away_def_rating'],
            "home_record": game['home_record'],
            "away_record": game['away_record'],
            "injuries": game['injuries'],
            "recent_form": game['recent_form'],
            "use_gpt": True,
            "monte_carlo_iterations": 10000
        }
    )

    if prediction_response.status_code != 200:
        raise HTTPException(500, "AI analysis failed")

    prediction = prediction_response.json()

    # Step 3: Get suggested questions
    suggestions_response = requests.post(
        f"{AI_API_URL}/api/v1/suggest-questions",
        json={"prediction": prediction}
    )

    suggestions = suggestions_response.json()

    # Step 4: Return to frontend
    return {
        "game_id": game_id,
        "prediction": prediction,
        "suggested_questions": suggestions.get('suggestions', [])
    }


@app.post("/api/chat/{game_id}")
async def chat_about_game(game_id: str, question: str, history: list = None):
    """
    Chat endpoint for follow-up questions.
    """

    # Get the prediction context for this game
    game_prediction = get_cached_prediction(game_id)

    # Format context for chat
    context = f"""
    Game: {game_prediction['away_team']} @ {game_prediction['home_team']}
    Prediction: {game_prediction['home_win_probability']:.1%} home win
    Analysis: {game_prediction['gpt_analysis']}
    Recommendation: {game_prediction['recommendation']}
    """

    # Send to AI chat
    chat_response = requests.post(
        f"{AI_API_URL}/api/v1/chat",
        json={
            "question": question,
            "game_context": context,
            "conversation_history": history or []
        }
    )

    return chat_response.json()


def fetch_game_from_database(game_id: str):
    """
    Fetch game data from your database or ESPN API.
    You implement this based on your data source.
    """
    # Example: Fetch from your database
    # game = db.query(Game).filter_by(id=game_id).first()

    # Or fetch from ESPN
    from src.data.espn_client import ESPNClient
    espn = ESPNClient()
    games = espn.get_todays_games("nba")
    game = games[0]  # Find your game

    # Calculate stats from recent games
    home_stats = calculate_team_stats(game['home_team_id'])
    away_stats = calculate_team_stats(game['away_team_id'])

    return {
        "home_team": game['home_team_name'],
        "away_team": game['away_team_name'],
        "sport": "nba",
        "home_ppg": home_stats['ppg'],
        "away_ppg": away_stats['ppg'],
        "home_def_rating": home_stats['def_rating'],
        "away_def_rating": away_stats['def_rating'],
        "home_record": game['home_record'],
        "away_record": game['away_record'],
        "injuries": fetch_injuries(game),
        "recent_form": calculate_recent_form(game['home_team_id'], game['away_team_id'])
    }
```

---

### 2. Frontend Integration (React Example)

```jsx
// GameCard.jsx
import React, { useState } from 'react';

function GameCard({ game }) {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [question, setQuestion] = useState('');

  const analyzeGame = async () => {
    setLoading(true);

    try {
      const response = await fetch(`/api/analyze-game/${game.id}`, {
        method: 'POST'
      });

      const data = await response.json();
      setPrediction(data.prediction);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) return;

    // Add user message to chat
    const newHistory = [
      ...chatHistory,
      { role: 'user', content: question }
    ];
    setChatHistory(newHistory);

    try {
      const response = await fetch(`/api/chat/${game.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          history: chatHistory
        })
      });

      const data = await response.json();

      // Add AI response to chat
      setChatHistory([
        ...newHistory,
        { role: 'assistant', content: data.answer }
      ]);

      setQuestion('');
    } catch (error) {
      console.error('Chat failed:', error);
    }
  };

  return (
    <div className="game-card">
      {/* Game Info */}
      <div className="game-info">
        <h3>{game.away_team} @ {game.home_team}</h3>
        <p>{game.date} - {game.time}</p>

        <button
          onClick={analyzeGame}
          disabled={loading}
          className="analyze-button"
        >
          {loading ? '⏳ Analyzing...' : '🤖 AI Predict & Analyze'}
        </button>
      </div>

      {/* Prediction Display */}
      {prediction && (
        <div className="prediction-panel">
          <h4>🤖 AI Prediction</h4>

          <div className="probabilities">
            <div className="prob-bar">
              <span className="team">{prediction.home_team}</span>
              <div className="bar">
                <div
                  className="fill home"
                  style={{width: `${prediction.home_win_probability * 100}%`}}
                />
              </div>
              <span className="pct">
                {(prediction.home_win_probability * 100).toFixed(1)}%
              </span>
            </div>

            <div className="prob-bar">
              <span className="team">{prediction.away_team}</span>
              <div className="bar">
                <div
                  className="fill away"
                  style={{width: `${prediction.away_win_probability * 100}%`}}
                />
              </div>
              <span className="pct">
                {(prediction.away_win_probability * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          <div className="expected-score">
            <strong>Expected Score:</strong> {' '}
            {prediction.predicted_home_score.toFixed(1)} - {prediction.predicted_away_score.toFixed(1)}
          </div>

          <div className="confidence">
            <strong>Confidence:</strong> {' '}
            {(prediction.confidence * 100).toFixed(1)}%
          </div>

          <div className="recommendation">
            <strong>Recommendation:</strong> {' '}
            <span className="rec-badge">{prediction.recommendation}</span>
          </div>

          {prediction.gpt_analysis && (
            <div className="ai-analysis">
              <strong>🧠 AI Analysis:</strong>
              <p>{prediction.gpt_analysis}</p>
            </div>
          )}

          <button
            onClick={() => setChatOpen(!chatOpen)}
            className="chat-toggle"
          >
            💬 {chatOpen ? 'Hide' : 'Ask AI Questions'}
          </button>
        </div>
      )}

      {/* Chat Interface */}
      {chatOpen && prediction && (
        <div className="chat-panel">
          <div className="chat-messages">
            {chatHistory.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <strong>{msg.role === 'user' ? 'You' : 'AI'}:</strong>
                <p>{msg.content}</p>
              </div>
            ))}
          </div>

          <div className="suggested-questions">
            <p><strong>Suggested questions:</strong></p>
            {prediction.suggested_questions?.map((q, idx) => (
              <button
                key={idx}
                onClick={() => setQuestion(q)}
                className="suggested-q"
              >
                {q}
              </button>
            ))}
          </div>

          <div className="chat-input">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && askQuestion()}
              placeholder="Ask a question about this game..."
            />
            <button onClick={askQuestion}>Send</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default GameCard;
```

---

### 3. Styling (CSS)

```css
/* GameCard.css */
.game-card {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 20px;
  margin: 10px 0;
  background: white;
}

.analyze-button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 6px;
  font-size: 16px;
  cursor: pointer;
  margin-top: 10px;
}

.analyze-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.prediction-panel {
  margin-top: 20px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
}

.prob-bar {
  display: flex;
  align-items: center;
  margin: 10px 0;
  gap: 10px;
}

.prob-bar .bar {
  flex: 1;
  height: 30px;
  background: #e0e0e0;
  border-radius: 15px;
  overflow: hidden;
}

.prob-bar .fill {
  height: 100%;
  transition: width 0.5s ease;
}

.prob-bar .fill.home {
  background: linear-gradient(90deg, #667eea, #764ba2);
}

.prob-bar .fill.away {
  background: linear-gradient(90deg, #f093fb, #f5576c);
}

.recommendation {
  margin: 15px 0;
  padding: 10px;
  background: white;
  border-left: 4px solid #667eea;
}

.rec-badge {
  font-weight: bold;
  color: #667eea;
}

.chat-panel {
  margin-top: 20px;
  border-top: 2px solid #ddd;
  padding-top: 20px;
}

.chat-messages {
  max-height: 300px;
  overflow-y: auto;
  margin-bottom: 15px;
}

.message {
  margin: 10px 0;
  padding: 10px;
  border-radius: 6px;
}

.message.user {
  background: #e3f2fd;
  margin-left: 20px;
}

.message.assistant {
  background: #f5f5f5;
  margin-right: 20px;
}

.suggested-questions {
  margin: 15px 0;
}

.suggested-q {
  display: block;
  width: 100%;
  text-align: left;
  padding: 8px 12px;
  margin: 5px 0;
  background: white;
  border: 1px solid #ddd;
  border-radius: 4px;
  cursor: pointer;
}

.suggested-q:hover {
  background: #f0f0f0;
}

.chat-input {
  display: flex;
  gap: 10px;
}

.chat-input input {
  flex: 1;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.chat-input button {
  padding: 10px 20px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
```

---

## 🎯 Complete User Flow

### 1. User Browses Games
```
Your Website shows:
  ⚽ Lakers vs Celtics - Nov 15, 7:30 PM
  [🤖 AI Predict & Analyze]
```

### 2. User Clicks "Analyze"
```
Loading → AI analyzing...
  ↓
Shows prediction:
  Lakers: 45%  ████████░░
  Celtics: 55% ██████████

  Expected: 110-113
  Confidence: 72%
  Recommendation: LEAN AWAY - Celtics (55%)

  AI Analysis: "Celtics have strong momentum with a 4-game win streak.
  Lakers questionable with AD's ankle injury..."

  [💬 Ask AI Questions]
```

### 3. User Opens Chat
```
Suggested questions:
  • What factors favor the Celtics?
  • How confident should I be?
  • What could change the outcome?

[Ask a question...] [Send]
```

### 4. Interactive Conversation
```
User: "What if AD plays?"
AI: "If Anthony Davis plays, it could shift the probability by 5-8%
in the Lakers' favor. His defensive presence is crucial..."

User: "Should I bet on this?"
AI: "The 55% probability suggests a close game. Consider the value
vs the betting odds..."
```

---

## 📊 API Endpoints Summary

### Your Backend Calls:

1. **Analyze Game** → `POST /api/v1/predict`
   - Send: Game data with stats
   - Get: Full AI prediction

2. **Get Suggestions** → `POST /api/v1/suggest-questions`
   - Send: Prediction
   - Get: Suggested questions

3. **Chat** → `POST /api/v1/chat`
   - Send: Question + context
   - Get: AI answer

4. **Explain** → `POST /api/v1/explain`
   - Send: Prediction + aspect
   - Get: Detailed explanation

---

## 💰 Cost Estimate

**Per game analysis:**
- Prediction with GPT: ~$0.001
- Chat (3 questions): ~$0.0003
- **Total: ~$0.0013 per game**

**For 1000 games/day:**
- ~$1.30/day
- ~$39/month

**Optimization:**
- Cache predictions: Reuse for same game
- Batch similar games
- Use GPT-4o-mini (already configured)

---

## 🚀 Next Steps

1. **Start the AI API:**
   ```bash
   ./start_api.sh
   ```

2. **Test with curl:**
   ```bash
   python test_api.py
   ```

3. **Integrate into your backend**

4. **Build the frontend UI**

5. **Deploy both services**

---

## 🎨 UI/UX Tips

- Show loading animation during analysis
- Display probabilities as progress bars
- Highlight key insights from AI
- Make chat feel conversational
- Show suggested questions prominently
- Cache predictions to avoid redundant API calls
- Add share/save prediction feature

---

This is a complete, production-ready integration! Your users get:
✅ On-demand AI predictions
✅ Interactive chat interface
✅ Visual probability displays
✅ Suggested questions
✅ Full game analysis

All powered by Monte Carlo + GPT! 🚀
