"""
Application configuration management.

Loads settings from environment variables with sensible defaults for the
OmniBrain agentic orchestrator. Supports both OpenAI and Google Gemini APIs.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
import openai


class Settings(BaseSettings):
    """Core application settings initialized from environment or defaults."""
    
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    QDRANT_URL: str = "http://localhost:6333"
    DATABASE_URL: str = "sqlite:///./data/stock.db"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    LLM_MODEL: str = "gpt-4o"
    GUARDRAIL_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RELEVANCE_THRESHOLD: float = 0.65
    MAX_CORRECTION_ATTEMPTS: int = 2
    
    DATA_DIR: Path = Path("data")
    EXTRACTED_IMAGES_DIR: Path = Path("data/extracted_images")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        
    def ensure_directories(self) -> None:
        """Create required application directories."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_gemini(self) -> bool:
        key = self.GEMINI_API_KEY or self.OPENAI_API_KEY
        return bool(self.GEMINI_API_KEY or key.startswith("AQ.") or key.startswith("AIzaSy"))

    @property
    def effective_api_key(self) -> str:
        return self.GEMINI_API_KEY or self.OPENAI_API_KEY

    @property
    def effective_base_url(self) -> str | None:
        if self.OPENAI_BASE_URL:
            return self.OPENAI_BASE_URL
        if self.is_gemini:
            return "https://generativelanguage.googleapis.com/v1beta/openai/"
        return None

    @property
    def effective_llm_model(self) -> str:
        if self.is_gemini and (not self.LLM_MODEL or self.LLM_MODEL in ("gpt-4o", "gpt-4o-mini")):
            return "gemini-3.8-flash"
        return self.LLM_MODEL

    @property
    def effective_embedding_model(self) -> str:
        if self.is_gemini and (not self.EMBEDDING_MODEL or self.EMBEDDING_MODEL == "text-embedding-3-small"):
            return "gemini-embedding-001"
        return self.EMBEDDING_MODEL


try:
    import streamlit as st
    for _k, _v in getattr(st, "secrets", {}).items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:
    pass

settings = Settings()


def get_openai_client(**extra_kwargs) -> openai.OpenAI:
    """Return configured OpenAI client for either OpenAI or Gemini API."""
    kwargs = {"api_key": settings.effective_api_key, "max_retries": 0}
    if settings.effective_base_url:
        kwargs["base_url"] = settings.effective_base_url
    kwargs.update(extra_kwargs)
    return openai.OpenAI(**kwargs)
