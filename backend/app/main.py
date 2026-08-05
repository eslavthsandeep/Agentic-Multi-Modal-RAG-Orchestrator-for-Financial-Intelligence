"""FastAPI entrypoint for the OmniBrain backend.

Configures routing, CORS, and startup/shutdown lifecycle. On startup
it ensures data directories exist, Qdrant collections are created, and
the SQLite schema is initialized.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes_upload import router as upload_router
from app.api.routes_query import router as query_router
from app.api.routes_status import router as status_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown — fail fast if core infra is missing."""
    logger.info("Starting OmniBrain backend")
    settings.ensure_directories()

    # Data sub-directories used at runtime
    (settings.DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
    (settings.DATA_DIR / "sample_pdfs").mkdir(parents=True, exist_ok=True)

    try:
        from app.db.qdrant_client import ensure_collections
        from app.db.sql_client import init_tables

        ensure_collections()
        init_tables()
        logger.info("Database collections and tables initialized")
    except Exception as exc:
        # Log but don't crash — Qdrant might not be running during tests
        logger.warning("Database init incomplete (non-fatal): %s", exc)

    yield
    logger.info("Shutting down OmniBrain backend")


app = FastAPI(
    title="OmniBrain API",
    description="Agentic Multi-Modal RAG Orchestrator",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(query_router)
app.include_router(status_router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"service": "omnibrain", "status": "running"}
