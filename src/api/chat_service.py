"""
Chat service for interactive Q&A about game predictions.
"""

from typing import List, Dict, Optional
from openai import OpenAI
import os
from loguru import logger


class ChatService:
    """
    Chat service for answering questions about game predictions.
    """

    def __init__(self):
        """Initialize chat service."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not configured")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    def chat(
        self,
        user_question: str,
        game_context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict:
        """
        Answer user questions about a game.

        Args:
            user_question: User's question
            game_context: Previous prediction/analysis context
            conversation_history: Previous messages in conversation

        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Chat question: {user_question[:100]}...")

        # Build messages
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            }
        ]

        # Add game context if provided
        if game_context:
            messages.append({
                "role": "assistant",
                "content": f"Here's my analysis of this game:\n\n{game_context}"
            })

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add user question
        messages.append({
            "role": "user",
            "content": user_question
        })

        # Get response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )

            answer = response.choices[0].message.content

            return {
                "question": user_question,
                "answer": answer,
                "tokens_used": response.usage.total_tokens,
                "model": self.model
            }

        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return {
                "question": user_question,
                "answer": f"I encountered an error: {str(e)}",
                "error": True
            }

    def _get_system_prompt(self) -> str:
        """Get system prompt for chat."""
        return """You are an expert sports analyst AI assistant.

Your role is to:
- Answer questions about sports games and predictions
- Explain analysis and reasoning
- Provide insights on teams, players, and matchups
- Discuss betting strategies and risk management
- Be honest about uncertainty

Guidelines:
- Keep responses concise and clear (2-3 paragraphs max)
- Use specific examples and data when available
- Acknowledge limitations of predictions
- Focus on educational value
- Don't make guarantees about outcomes

You have access to the game analysis provided in the conversation context."""

    def explain_prediction(
        self,
        prediction_data: Dict,
        aspect: str = "general"
    ) -> str:
        """
        Explain a specific aspect of the prediction.

        Args:
            prediction_data: The full prediction dictionary
            aspect: What to explain (general, probability, gpt, monte_carlo)

        Returns:
            Explanation text
        """
        prompts = {
            "general": "Explain this prediction in simple terms for someone new to sports betting.",
            "probability": "Explain how the win probability was calculated and what it means.",
            "gpt": "Explain the qualitative factors (GPT analysis) that influenced this prediction.",
            "monte_carlo": "Explain how the Monte Carlo simulation works and why it's used.",
            "confidence": "Explain the confidence level and what factors affect it.",
            "recommendation": "Explain the betting recommendation and why it was given."
        }

        prompt = prompts.get(aspect, prompts["general"])

        context = f"""
Game: {prediction_data.get('away_team')} @ {prediction_data.get('home_team')}
Home Win Probability: {prediction_data.get('home_win_probability', 0):.1%}
Away Win Probability: {prediction_data.get('away_win_probability', 0):.1%}
Expected Score: {prediction_data.get('predicted_home_score'):.1f} - {prediction_data.get('predicted_away_score'):.1f}
Confidence: {prediction_data.get('confidence', 0):.1%}
Recommendation: {prediction_data.get('recommendation')}
GPT Analysis: {prediction_data.get('gpt_analysis', 'N/A')}
Models Used: {', '.join(prediction_data.get('models_used', []))}

{prompt}
"""

        response = self.chat(
            user_question=context,
            game_context=None
        )

        return response['answer']

    def suggest_questions(
        self,
        prediction_data: Dict
    ) -> List[str]:
        """
        Suggest relevant questions users might ask about this prediction.

        Args:
            prediction_data: The prediction dictionary

        Returns:
            List of suggested questions
        """
        # Default suggestions based on prediction
        suggestions = [
            "What factors most influenced this prediction?",
            "How confident should I be in this prediction?",
            "What could change the outcome?",
        ]

        # Add context-specific suggestions
        home_prob = prediction_data.get('home_win_probability', 0.5)

        if home_prob > 0.65:
            suggestions.append(f"Why is {prediction_data.get('home_team')} favored?")
        elif home_prob < 0.35:
            suggestions.append(f"Why is {prediction_data.get('away_team')} favored?")
        else:
            suggestions.append("Why is this game so close?")

        if prediction_data.get('gpt_analysis'):
            suggestions.append("Tell me more about the GPT analysis")

        if prediction_data.get('confidence', 0) < 0.6:
            suggestions.append("Why is the confidence low?")

        return suggestions[:5]  # Return top 5


# Global instance
chat_service = ChatService()
