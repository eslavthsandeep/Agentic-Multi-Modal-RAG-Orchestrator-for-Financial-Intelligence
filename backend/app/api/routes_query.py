"""Query endpoint that routes user questions through the LangGraph pipeline.

Validates the document exists and is ready, then delegates to the supervisor
graph. The graph itself is synchronous LangGraph but exposes an async
interface, so we call it directly with await.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.routes_upload import _document_status
from app.agents.supervisor import run_query

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    document_id: str
    query: str
    chat_history: list[dict] = []


@router.post("/api/query")
async def query_document(request: QueryRequest):
    """Ask a question about an uploaded document.

    Routes through the LangGraph supervisor which dispatches to search,
    SQL, and/or vision agents, then synthesizes a cited answer.
    """
    doc_status = _document_status.get(request.document_id)
    if not doc_status:
        raise HTTPException(status_code=404, detail="Document not found. Upload it first.")

    if doc_status.get("status") == "processing":
        raise HTTPException(status_code=409, detail="Document is still processing. Try again shortly.")

    if doc_status.get("status") == "failed":
        raise HTTPException(status_code=422, detail="Document ingestion failed. Re-upload required.")

    try:
        result = await run_query(
            query=request.query,
            document_id=request.document_id,
            chat_history=request.chat_history,
        )
        return result

    except Exception as exc:
        logger.error("Query failed for document %s: %s", request.document_id, exc)
        raise HTTPException(status_code=500, detail="Query processing failed") from exc
