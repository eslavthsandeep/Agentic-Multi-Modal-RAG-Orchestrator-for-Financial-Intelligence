"""
Application configuration management.

Loads settings from environment variables with sensible defaults for the
OmniBrain agentic orchestrator.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Core application settings initialized from environment or defaults."""
    
    OPENAI_API_KEY: str = ""
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
        
    def ensure_directories(self) -> None:
        """Create required application directories."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
