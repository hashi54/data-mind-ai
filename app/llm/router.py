import json
from typing import Dict, Any, Tuple
from app.llm.client import llm_client
from app.llm.prompts import ROUTER_SYSTEM_PROMPT
from app.core.logging import logger


class IntentRouter:
    """Classifies user inquiries to route them to the appropriate specialized agent."""

    @staticmethod
    def route_intent(question: str) -> Tuple[str, str]:
        """
        Determines the destination agent for the question.
        Returns (intent_type, reasoning).
        """
        try:
            response_text = llm_client.generate_response(
                system_prompt=ROUTER_SYSTEM_PROMPT,
                user_prompt=question,
            )
            # Clean JSON fences if present
            cleaned = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            intent = data.get("intent", "SQL_QUERY").upper()
            reasoning = data.get("reasoning", "Routed by intent engine.")
            logger.info(f"Routed question '{question[:60]}' -> Intent: {intent}")
            return intent, reasoning
        except Exception as e:
            logger.warning(f"Error parsing router LLM response: {e}. Falling back to keyword router.")
            return IntentRouter._fallback_router(question)

    @staticmethod
    def _fallback_router(question: str) -> Tuple[str, str]:
        q = question.lower()
        if any(k in q for k in ["refund policy", "travel policy", "handbook", "specs", "warranty", "catalog"]) and any(k in q for k in ["how much", "lost", "total revenue", "orders"]):
            return "HYBRID_SQL_RAG", "Question connects policy criteria with live database aggregations."
        if any(k in q for k in ["policy", "guidelines", "handbook", "warranty", "aerobook", "catalog", "allowance", "leave"]):
            return "RAG_DOCS", "Knowledge base document search required."
        if any(k in q for k in ["churn", "predict", "forecast", "future", "likely to leave", "risk", "clv", "contact this week"]):
            return "ML_PREDICTION", "Predictive ML inference pipeline required."
        if any(k in q for k in ["profile", "eda", "health score", "missing", "correlation"]):
            return "EDA_ANALYSIS", "Automated exploratory data analysis required."
        if any(k in q for k in ["sales", "revenue", "orders", "kerala", "august", "profit", "margin", "top product", "why did"]):
            return "SQL_QUERY", "Relational database SQL analytics required."
        return "GENERAL_CHAT", "General conversational business assistant."
