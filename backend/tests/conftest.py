"""
Shared pytest fixtures for OmniBrain test suite.
Provides API client, sample PDFs, and service mocks so tests run without real keys.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure no real API keys are used during tests
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client wired to the real app instance."""
    return TestClient(app)


@pytest.fixture
def sample_pdf_path():
    """Generate a minimal valid PDF with text and yield its path.

    Uses pymupdf to create a real PDF so the parser can extract text
    from it — a raw bytes stub won't exercise the actual parsing logic.
    """
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "OmniBrain test document.")
    page.insert_text((72, 100), "Revenue grew 12% year-over-year in Q3 2024.")
    page.insert_text((72, 128), "Operating expenses were $4.2 billion.")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = tmp_file.name
    tmp_file.close()

    doc.save(pdf_path)
    doc.close()

    yield pdf_path
    Path(pdf_path).unlink(missing_ok=True)


@pytest.fixture
def mock_openai(monkeypatch):
    """Patch OpenAI client so no real API calls are made during tests."""
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.01] * 1536

    mock_response = MagicMock()
    mock_response.data = [mock_embedding]

    mock_chat_choice = MagicMock()
    mock_chat_choice.message.content = '{"route": "search", "reasoning": "test"}'

    mock_chat_response = MagicMock()
    mock_chat_response.choices = [mock_chat_choice]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response
    mock_client.chat.completions.create.return_value = mock_chat_response

    with patch("openai.OpenAI", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_qdrant(monkeypatch):
    """Patch Qdrant client so tests don't need a running Qdrant instance."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])

    mock_query_result = MagicMock()
    mock_query_result.points = []
    mock_client.query_points.return_value = mock_query_result

    with patch("app.db.qdrant_client.get_qdrant_client", return_value=mock_client):
        yield mock_client
