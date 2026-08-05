"""Integration with Langfuse for observability and tracing.
Provides decorators and client instances to track LLM interactions and agent latency."""

import functools
import logging
import time

from app.config import settings

logger = logging.getLogger(__name__)

_langfuse_instance = None
_langfuse_available = False

def get_langfuse():
    """Get the Langfuse client. Returns None if not configured."""
    global _langfuse_instance, _langfuse_available
    if _langfuse_instance is not None:
        return _langfuse_instance
    
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.info("Langfuse keys not configured — tracing disabled")
        _langfuse_available = False
        return None
    
    try:
        from langfuse import Langfuse
        _langfuse_instance = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        _langfuse_available = True
        logger.info("Langfuse tracing initialized")
        return _langfuse_instance
    except Exception as exc:
        logger.warning("Failed to initialize Langfuse: %s", exc)
        _langfuse_available = False
        return None

def traced(agent_name: str):
    """Decorator that logs agent execution to Langfuse when available.
    
    Falls back to local timing logs when Langfuse isn't configured,
    so agents always get performance visibility regardless of setup.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            lf = get_langfuse()
            
            try:
                # Manual trace initialization instead of the @observe wrapper to offer more fine-grained control
                trace = None
                span = None
                if lf:
                    trace = lf.trace(name=f"{agent_name}_execution")
                    span = trace.span(name=agent_name, start_time=start)
                    
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                if span:
                    span.end(end_time=time.perf_counter())
                    
                logger.info("%s completed in %.2fs", agent_name, elapsed)
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                logger.error("%s failed after %.2fs: %s", agent_name, elapsed, exc)
                raise
        return wrapper
    return decorator
