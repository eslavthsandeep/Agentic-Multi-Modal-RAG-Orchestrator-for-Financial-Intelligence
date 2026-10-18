"""Input and output semantic guardrails for query validation.
Wraps NeMo Guardrails to ensure queries are on-topic and answers are grounded."""

from __future__ import annotations

import logging
import os

try:
    from nemoguardrails import LLMRails, RailsConfig
    NEMO_AVAILABLE = True
except ImportError:
    LLMRails = None
    RailsConfig = None
    NEMO_AVAILABLE = False

logger = logging.getLogger(__name__)

_rails_instance = None
_config_path = os.path.join(os.path.dirname(__file__), "nemo_config")

def _get_rails():
    """Load NeMo rails config. Cached after first call."""
    global _rails_instance
    if _rails_instance is not None:
        return _rails_instance
    if not NEMO_AVAILABLE:
        logger.info("NeMo Guardrails not installed; using built-in query validator")
        return None
        
    try:
        config = RailsConfig.from_path(_config_path)
        _rails_instance = LLMRails(config)
        logger.info("NeMo Guardrails configuration loaded successfully")
        return _rails_instance
    except Exception as exc:
        logger.warning("Failed to load NeMo Guardrails config: %s", exc)
        return None

async def check_input(query: str, document_context: str = "") -> dict:
    """Validate that a user query is on-topic for the uploaded document."""
    q_clean = query.strip(' \t\n\r"\'').lower()

    # Fast heuristic check for obvious off-topic prompts
    off_topic_patterns = [
        "machine learning", "meachine learning", "deep learning", "neural network",
        "python script", "write code", "sort a list", "recipe", "how to cook",
        "tell me a joke", "write a poem", "who won", "movie recommendation",
        "biology", "chemistry", "quantum physics", "weather in", "translate to",
        "give the types", "what is machine learning", "what is meachine learning"
    ]
    for pattern in off_topic_patterns:
        if pattern in q_clean:
            return {
                "passed": False,
                "reason": "This question is outside the scope of financial document analysis. OmniBrain only answers questions regarding the uploaded financial filings and stock data."
            }

    rails = _get_rails()
    if rails:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": query}])
            if "I cannot answer" in response or "off-topic" in response.lower():
                return {"passed": False, "reason": "Query rejected by input guardrails"}
            return {"passed": True, "reason": None}
        except Exception as exc:
            logger.error("Input guardrail check failed: %s", exc)

    # Use LLM for semantic validation if API key is present
    try:
        from app.config import settings, get_openai_client
        if settings.effective_api_key:
            client = get_openai_client()
            val_prompt = (
                "You are an input guardrail for a financial document QA system. "
                "Determine if the user query is about financial reports, corporate filings, company business, executive management, or stock data.\n"
                f"Query: \"{query}\"\n"
                "Answer ONLY with 'ON_TOPIC' or 'OFF_TOPIC'."
            )
            res = client.chat.completions.create(
                model=settings.effective_llm_model,
                messages=[{"role": "user", "content": val_prompt}],
                max_tokens=5,
                temperature=0.0
            )
            decision = (res.choices[0].message.content or "").strip().upper()
            if "OFF_TOPIC" in decision:
                return {
                    "passed": False,
                    "reason": "This question is outside the scope of financial document analysis. OmniBrain only answers questions regarding the uploaded financial filings and stock data."
                }
    except Exception as e:
        logger.warning("LLM guardrail check skipped: %s", e)

    return {"passed": True, "reason": None}

async def check_output(answer: str, retrieved_context: str) -> dict:
    """Validate that the generated answer is grounded in retrieved context."""
    rails = _get_rails()
    if not rails:
        return {"passed": True, "reason": None}
        
    try:
        prompt = f"Context: {retrieved_context}\nAnswer: {answer}"
        response = await rails.generate_async(messages=[{"role": "user", "content": prompt}])
        
        if "unsupported" in response.lower() or "not grounded" in response.lower():
            return {"passed": False, "reason": "Answer not grounded in context"}
            
        return {"passed": True, "reason": None}
    except Exception as exc:
        logger.error("Output guardrail check failed: %s", exc)
        return {"passed": True, "reason": "Guardrail evaluation error"}
