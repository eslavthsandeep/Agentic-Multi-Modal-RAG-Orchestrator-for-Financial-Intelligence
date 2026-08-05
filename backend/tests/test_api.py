"""
API endpoint integration tests.
Uses FastAPI TestClient with mocked external services so tests
run without Qdrant, OpenAI, or any other live infrastructure.
"""

import io
from unittest.mock import patch, MagicMock

import pytest


class TestHealthEndpoints:

    def test_root_health_check(self, client):
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "omnibrain"
        assert body["status"] == "running"

    def test_system_status_shape(self, client):
        response = client.get("/api/agents/status")
        assert response.status_code == 200
        body = response.json()
        # Should report on all three subsystems
        assert "qdrant" in body
        assert "sql_db" in body
        assert "langfuse" in body


class TestUploadEndpoint:

    def test_upload_returns_processing_status(self, client):
        """Uploading a PDF should return a document_id and 'processing' status."""
        fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")
        response = client.post(
            "/api/upload",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
        )
        assert response.status_code == 200
        body = response.json()
        assert "document_id" in body
        assert body["status"] == "processing"

    def test_upload_status_unknown_document(self, client):
        """Polling status for a non-existent document should indicate not found."""
        response = client.get("/api/upload/nonexistent_doc_999/status")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "not_found"


class TestQueryEndpoint:

    def test_query_without_document_id(self, client):
        """Querying without a valid document should still return a response (may be an error)."""
        response = client.post(
            "/api/query",
            json={
                "document_id": "nonexistent_doc",
                "query": "What is revenue?",
                "chat_history": [],
            },
        )
        # The endpoint should handle this gracefully
        assert response.status_code in (200, 404, 422)
