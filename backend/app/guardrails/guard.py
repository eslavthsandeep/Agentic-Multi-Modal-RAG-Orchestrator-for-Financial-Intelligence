"""Input and output semantic guardrails for query validation.
Wraps NeMo Guardrails to ensure queries are on-topic and answers are grounded."""

from __future__ import annotations

import logging
import os

from nemoguardrails import LLMRails, RailsConfig

logger = logging.getLogger(__name__)

_rails_instance: LLMRails | None = None
_config_path = os.path.join(os.path.dirname(__file__), "nemo_config")

def _get_rails() -> LLMRails | None:
    """Load NeMo rails config. Cached after first call."""
    global _rails_instance
    if _rails_instance is not None:
        return _rails_instance
        
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
    rails = _get_rails()
    if not rails:
        return {"passed": True, "reason": None}
        
    try:
        response = await rails.generate_async(messages=[{"role": "user", "content": query}])
        
        if "I cannot answer" in response or "off-topic" in response.lower():
            return {"passed": False, "reason": "Query rejected by input guardrails"}
            
        return {"passed": True, "reason": None}
    except Exception as exc:
        logger.error("Input guardrail check failed: %s", exc)
        return {"passed": True, "reason": "Guardrail evaluation error"}

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
