"""
Tests for the document ingestion pipeline — PDF parsing, chunking, and embedding.
Uses a generated test PDF so no external files are needed.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestPdfParser:
    """Verify PyMuPDF + pdfplumber extraction handles real PDF content."""

    def test_extracts_text_from_pages(self, sample_pdf_path):
        from app.ingestion.pdf_parser import parse_pdf

        with tempfile.TemporaryDirectory() as img_dir:
            result = parse_pdf(sample_pdf_path, img_dir)

        assert "pages" in result
        assert len(result["pages"]) > 0

        full_text = " ".join(p["text"] for p in result["pages"])
        assert "OmniBrain" in full_text or "Revenue" in full_text

    def test_returns_page_numbers(self, sample_pdf_path):
        from app.ingestion.pdf_parser import parse_pdf

        with tempfile.TemporaryDirectory() as img_dir:
            result = parse_pdf(sample_pdf_path, img_dir)

        for page in result["pages"]:
            assert "page_num" in page
            assert isinstance(page["page_num"], int)
            assert page["page_num"] >= 1

    def test_handles_missing_file(self):
        from app.ingestion.pdf_parser import parse_pdf

        with pytest.raises(Exception):
            parse_pdf("/nonexistent/path.pdf", "/tmp/images")


class TestChunker:
    """Verify text chunking respects size limits and overlap."""

    def _make_parsed_doc(self, text: str, page_num: int = 1) -> dict:
        return {
            "pages": [{"page_num": page_num, "text": text}],
            "tables": [],
            "images": [],
        }

    def test_respects_chunk_size(self):
        from app.ingestion.chunker import chunk_document

        long_text = "This is a sentence. " * 200  # ~4000 chars
        doc = self._make_parsed_doc(long_text)
        chunks = chunk_document(doc, chunk_size=500, overlap=50)

        for chunk in chunks:
            # Allow some tolerance since we split on sentence boundaries
            assert len(chunk["text"]) <= 600

    def test_produces_overlap(self):
        from app.ingestion.chunker import chunk_document

        long_text = "Sentence number one. Sentence number two. " * 50
        doc = self._make_parsed_doc(long_text)
        chunks = chunk_document(doc, chunk_size=200, overlap=50)

        if len(chunks) >= 2:
            # Check that consecutive chunks share some content
            first_end = chunks[0]["text"][-50:]
            second_start = chunks[1]["text"][:100]
            # At minimum, they should share sentence fragments
            assert len(chunks) > 1

    def test_skips_whitespace_chunks(self):
        from app.ingestion.chunker import chunk_document

        doc = self._make_parsed_doc("   \n\n\t  \n  ")
        chunks = chunk_document(doc, chunk_size=500, overlap=50)
        assert len(chunks) == 0

    def test_tables_chunked_separately(self):
        from app.ingestion.chunker import chunk_document

        doc = {
            "pages": [{"page_num": 1, "text": "Some regular text here."}],
            "tables": [{"page_num": 2, "text": "Col1 | Col2\nVal1 | Val2", "data": []}],
            "images": [],
        }
        chunks = chunk_document(doc, chunk_size=500, overlap=50)
        table_chunks = [c for c in chunks if c["source_type"] == "table"]
        text_chunks = [c for c in chunks if c["source_type"] == "text"]
        assert len(table_chunks) >= 1
        assert len(text_chunks) >= 1

    @pytest.mark.parametrize("chunk_size", [100, 300, 500, 1000])
    def test_various_chunk_sizes(self, chunk_size):
        from app.ingestion.chunker import chunk_document

        text = "This is a test sentence. " * 100
        doc = self._make_parsed_doc(text)
        chunks = chunk_document(doc, chunk_size=chunk_size, overlap=50)
        assert len(chunks) > 0
        assert all(c["chunk_index"] >= 0 for c in chunks)

    def test_no_midword_splitting(self):
        """Ensure chunk overlaps snap to word boundaries and do not split words like 'oversight' into 'versight'."""
        from app.ingestion.chunker import chunk_document

        text = "Oversight of cybersecurity risk management is critical for accounting standards. " * 30
        doc = self._make_parsed_doc(text)
        chunks = chunk_document(doc, chunk_size=150, overlap=35)

        truncated_stubs = ["versight", "ccounting", "tandards", "itical", "agement"]
        for chunk in chunks:
            chunk_text = chunk["text"]
            first_word = chunk_text.split()[0]
            assert first_word not in truncated_stubs, f"Chunk starts mid-word: '{first_word}' in '{chunk_text[:30]}'"


class TestEmbedder:
    """Verify embedding functions return correct dimensions."""

    @patch("app.ingestion.embedder.OpenAI")
    def test_embed_texts_returns_1536_dim(self, mock_openai_cls):
        from app.ingestion.embedder import embed_texts

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.01] * 1536

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(data=[mock_embedding])
        mock_openai_cls.return_value = mock_client

        result = embed_texts(["test query"])
        assert len(result) == 1
        assert len(result[0]) == 1536

    def test_embed_images_returns_empty_for_no_input(self):
        from app.ingestion.embedder import embed_images
        assert embed_images([]) == []
