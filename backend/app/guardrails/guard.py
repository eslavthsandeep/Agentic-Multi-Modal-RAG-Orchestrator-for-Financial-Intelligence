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

def is_off_topic_heuristic(query: str) -> tuple[bool, str]:
    """Fast deterministic heuristic to catch common off-topic queries."""
    import re
    q = query.strip().lower()

    # Financial exceptions: if these are present, let it pass to financial reasoning
    financial_keywords = [
        "stock", "share", "shares", "price", "volume", "trading", "ticker", "aapl", "nasdaq",
        "revenue", "sales", "net sales", "income", "profit", "margin", "gross margin", "tax",
        "effective tax rate", "provision", "cash flow", "balance sheet", "10-k", "10-q", "filing",
        "apple", "tim cook", "iphone", "mac", "ipad", "services", "wearables", "segment",
        "antitrust", "doj", "lawsuit", "repurchase", "buyback", "dividend"
    ]
    has_strong_financial = any(re.search(rf"\b{k}\b", q) for k in financial_keywords)

    # Patterns for off-topic categories
    ml_pattern = r"\b(m[a-z]{3,8}\s*learn[a-z]*|deep\s*learn|neural\s*net|artificial\s*intel|nlp|llm|gpt|chatgpt|transformer|supervised|unsupervised|reinforcement\s*learn)\b"
    politics_pattern = r"\b(pres[ie]dent\s*(of|in)?|prime\s*minister|white\s*house|congress|senat(or|e)|chancellor|governor\s+of|election\b)"
    sports_pattern = r"\b(fifa|world\s*cup|cricket|football|soccer|basketball|nba|nfl|tennis|super\s*bowl|olympics|messi|ronaldo)\b"
    trivia_pattern = r"\b(capital\s*of|weather\s*(in|today|tomorrow)|recipe|how\s*to\s*cook|cake|pizza|joke|poem|story|translate\s*to|horoscope|zodiac|actor|actress|movie)\b"
    coding_pattern = r"\b(write\s*(a\s*)?(python|javascript|c\+\+|java|code|script|function)|sort\s*a\s*list|fibonacci|binary\s*search)\b"

    if re.search(ml_pattern, q) and not has_strong_financial:
        return True, "This question is outside the scope of financial document analysis. OmniBrain only answers questions regarding the uploaded financial filings and stock data."
    if re.search(politics_pattern, q) and not has_strong_financial:
        return True, "This question is outside the scope of financial document analysis. OmniBrain only answers questions regarding the uploaded financial filings and stock data."
    if re.search(sports_pattern, q) and not has_strong_financial:
        return True, "This question is outside the scope of financial document analysis. OmniBrain only answers questions regarding the uploaded financial filings and stock data."
    if re.search(trivia_pattern, q) and not has_strong_financial:
        return True, "This question is outside the scope of financial document analysis. OmniBrain only answers questions regarding the uploaded financial filings and stock data."
    if re.search(coding_pattern, q) and not has_strong_financial:
        return True, "This question is outside the scope of financial document analysis. OmniBrain only answers questions regarding the uploaded financial filings and stock data."

    return False, ""


async def check_input(query: str, document_context: str = "") -> dict:
    """Validate that a user query is on-topic for the uploaded document."""
    # 1. Fast deterministic heuristic check
    is_off, reason = is_off_topic_heuristic(query)
    if is_off:
        return {"passed": False, "reason": reason}

    # 2. NeMo Guardrails check (if installed)
    rails = _get_rails()
    if rails:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": query}])
            if "I cannot answer" in response or "off-topic" in response.lower():
                return {"passed": False, "reason": "Query rejected by input guardrails"}
            return {"passed": True, "reason": None}
        except Exception as exc:
            logger.error("Input guardrail check failed: %s", exc)

    # 3. Semantic LLM validation for any queries that passed heuristics
    try:
        from app.config import settings, get_openai_client
        if settings.effective_api_key:
            client = get_openai_client()
            val_prompt = (
                "You are an input guardrail for OmniBrain, a financial document QA system for corporate SEC filings and stock data.\n"
                "Determine if the user query is about financial reports, corporate filings, company business, executive management, or stock data.\n"
                f"Query: \"{query}\"\n"
                "If the query is about general computer science, AI/ML concepts, world leaders, sports, entertainment, cooking, general coding, or unrelated trivia, answer OFF_TOPIC.\n"
                "If the query is about company performance, financials, accounting, SEC filings, or stock markets, answer ON_TOPIC.\n"
                "Answer ONLY with 'ON_TOPIC' or 'OFF_TOPIC'."
            )
            res = client.chat.completions.create(
                model=settings.effective_llm_model,
                messages=[{"role": "user", "content": val_prompt}],
                max_tokens=25,
                temperature=0.0
            )
            decision = (res.choices[0].message.content or "").strip().upper()
            if "OFF" in decision or "NOT" in decision:
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
