"""
GPT-based qualitative analysis for sports predictions.
Uses LLM to analyze contextual factors not captured by statistical models.
"""

from typing import Dict, List, Optional, Any
from openai import OpenAI
import os
import json
from loguru import logger
from datetime import datetime


class GPTAnalyzer:
    """
    GPT-powered qualitative analysis for sports predictions.

    Analyzes factors like:
    - Recent team form and momentum
    - Injury impacts
    - Head-to-head history and narratives
    - Situational factors (rest, travel, motivation)
    - Expert sentiment and betting trends
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize GPT analyzer.

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4o-mini for cost efficiency)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum response tokens
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyze_matchup(
        self,
        home_team: str,
        away_team: str,
        sport: str,
        game_date: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a matchup using GPT.

        Args:
            home_team: Home team name
            away_team: Away team name
            sport: Sport (e.g., 'NBA', 'NFL')
            game_date: Game date
            additional_context: Extra context (injuries, records, etc.)

        Returns:
            Analysis dictionary with insights and adjustment
        """
        prompt = self._build_analysis_prompt(
            home_team,
            away_team,
            sport,
            game_date,
            additional_context
        )

        try:
            logger.info(f"Analyzing {away_team} @ {home_team} with GPT")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(sport)
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )

            # Parse response
            result = json.loads(response.choices[0].message.content)

            # Add metadata
            result["model_used"] = self.model
            result["tokens_used"] = response.usage.total_tokens
            result["cost_estimate"] = self._estimate_cost(response.usage.total_tokens)

            logger.info(
                f"GPT analysis complete. Tokens: {response.usage.total_tokens}, "
                f"Confidence: {result.get('confidence', 'N/A')}"
            )

            return result

        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            return {
                "error": str(e),
                "analysis": "Unable to perform GPT analysis",
                "confidence": 0.0,
                "adjustment": 0.0,
            }

    def _get_system_prompt(self, sport: str) -> str:
        """Get system prompt tailored for sport."""
        return f"""You are an expert {sport} analyst with 20 years of experience.
Your specialty is identifying qualitative factors that statistical models miss, such as:
- Team momentum and recent form
- Key player injuries and their impact
- Motivational factors (rivalry games, revenge narratives, playoff implications)
- Situational factors (rest days, travel, back-to-back games)
- Coaching matchups and strategic advantages
- Intangible factors (team chemistry, locker room issues)

Provide objective, data-informed analysis. Be honest about uncertainty.
Your analysis will be combined with quantitative models to make predictions."""

    def _build_analysis_prompt(
        self,
        home_team: str,
        away_team: str,
        sport: str,
        game_date: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build analysis prompt with all available context."""
        prompt = f"""Analyze this {sport} matchup:

**Game:** {away_team} @ {home_team}
**Date:** {game_date}
"""

        if context:
            if "home_record" in context:
                prompt += f"\n**Home Team Record:** {context['home_record']}"
            if "away_record" in context:
                prompt += f"\n**Away Team Record:** {context['away_record']}"
            if "injuries" in context:
                prompt += f"\n**Injuries:** {context['injuries']}"
            if "recent_form" in context:
                prompt += f"\n**Recent Form:** {context['recent_form']}"
            if "head_to_head" in context:
                prompt += f"\n**Head-to-Head:** {context['head_to_head']}"
            if "betting_trends" in context:
                prompt += f"\n**Betting Trends:** {context['betting_trends']}"

        prompt += """

Provide your analysis in JSON format with the following structure:
{
  "team_form_analysis": "Analysis of recent team performance and momentum",
  "injury_impact": "Assessment of injury impacts on both teams",
  "situational_factors": "Rest, travel, motivation, and other situational factors",
  "key_matchups": "Important player or strategic matchups",
  "narrative_factors": "Rivalry, revenge, or other storylines",
  "edge_identified": "Which team has qualitative edge (home/away/neutral)",
  "confidence": 0.75,
  "adjustment": -0.03,
  "reasoning": "Brief explanation of the adjustment",
  "final_recommendation": "Overall assessment and betting recommendation"
}

**confidence**: Your confidence in this analysis (0.0 to 1.0)
**adjustment**: Suggested probability adjustment to statistical model (-0.10 to +0.10)
  - Positive adjustment favors home team
  - Negative adjustment favors away team
  - Use conservatively (typical range: -0.05 to +0.05)
"""

        return prompt

    def batch_analyze(
        self,
        games: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple games efficiently.

        Uses batch processing to reduce costs.

        Args:
            games: List of game dictionaries with required fields

        Returns:
            List of analysis results
        """
        results = []

        # Process in batch of 10 for efficiency
        batch_size = 10
        for i in range(0, len(games), batch_size):
            batch = games[i:i + batch_size]

            # Build batch prompt
            batch_prompt = self._build_batch_prompt(batch)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert sports analyst. "
                                      "Analyze multiple games efficiently."
                        },
                        {
                            "role": "user",
                            "content": batch_prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens * len(batch),
                    response_format={"type": "json_object"}
                )

                # Parse batch response
                batch_results = json.loads(response.choices[0].message.content)

                results.extend(batch_results.get("games", []))

                logger.info(
                    f"Batch analyzed {len(batch)} games. "
                    f"Tokens: {response.usage.total_tokens}"
                )

            except Exception as e:
                logger.error(f"Batch analysis failed: {e}")
                # Add error placeholders
                results.extend([{
                    "error": str(e),
                    "adjustment": 0.0,
                    "confidence": 0.0
                }] * len(batch))

        return results

    def _build_batch_prompt(self, games: List[Dict[str, Any]]) -> str:
        """Build prompt for batch game analysis."""
        prompt = "Analyze these games concisely:\n\n"

        for i, game in enumerate(games, 1):
            prompt += f"{i}. {game['away_team']} @ {game['home_team']}\n"
            if "context" in game:
                prompt += f"   Context: {json.dumps(game['context'])}\n"
            prompt += "\n"

        prompt += """
Return JSON with format:
{
  "games": [
    {
      "game_id": 1,
      "edge": "home/away/neutral",
      "confidence": 0.5,
      "adjustment": 0.0,
      "key_factor": "Brief key insight"
    },
    ...
  ]
}

Keep responses concise to save tokens."""

        return prompt

    def _estimate_cost(self, tokens: int) -> float:
        """
        Estimate cost for API call.

        GPT-4o-mini pricing (as of 2025):
        - $0.150 per 1M input tokens
        - $0.600 per 1M output tokens
        Approximating 50/50 split
        """
        avg_cost_per_token = (0.150 + 0.600) / 2 / 1_000_000
        return tokens * avg_cost_per_token

    def get_quick_insight(
        self,
        home_team: str,
        away_team: str,
        sport: str
    ) -> str:
        """
        Get quick one-sentence insight about the matchup.

        Optimized for minimal tokens.

        Args:
            home_team: Home team name
            away_team: Away team name
            sport: Sport

        Returns:
            One sentence insight
        """
        prompt = f"In one sentence, what's the key factor for {away_team} @ {home_team} in {sport}?"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a {sport} expert. Be concise."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=50,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Quick insight failed: {e}")
            return "No insight available"


# Convenience function
def analyze_nba_game(
    home_team: str,
    away_team: str,
    home_record: str = "",
    away_record: str = "",
    injuries: str = ""
) -> Dict[str, Any]:
    """Quick function to analyze an NBA game."""
    analyzer = GPTAnalyzer()

    context = {
        "home_record": home_record,
        "away_record": away_record,
        "injuries": injuries,
    }

    return analyzer.analyze_matchup(
        home_team=home_team,
        away_team=away_team,
        sport="NBA",
        game_date=datetime.now().strftime("%Y-%m-%d"),
        additional_context=context
    )


if __name__ == "__main__":
    # Test GPT analyzer (requires OPENAI_API_KEY)
    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable to test")
        exit(1)

    print("\n=== GPT Analyzer Example ===\n")

    analyzer = GPTAnalyzer()

    # Analyze Lakers vs Celtics
    analysis = analyzer.analyze_matchup(
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        sport="NBA",
        game_date="2024-11-13",
        additional_context={
            "home_record": "7-4",
            "away_record": "9-2",
            "injuries": "Lakers: Anthony Davis (questionable - ankle)",
            "recent_form": "Celtics won last 3 games, Lakers lost 2 of last 3"
        }
    )

    print("GPT Analysis:")
    print(f"  Edge: {analysis.get('edge_identified', 'N/A')}")
    print(f"  Confidence: {analysis.get('confidence', 0):.1%}")
    print(f"  Adjustment: {analysis.get('adjustment', 0):+.2%}")
    print(f"\n  Reasoning: {analysis.get('reasoning', 'N/A')}")
    print(f"\n  Cost: ${analysis.get('cost_estimate', 0):.6f}")
    print(f"  Tokens: {analysis.get('tokens_used', 0)}")
