from __future__ import annotations

import logging
from fastapi import APIRouter

from app.config import settings
from app.db.qdrant_client import get_qdrant_client
from app.db.sql_client import execute_query

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/api/agents/status")
async def system_status():
    """Check system health for connected services."""
    status_report = {}
    
    # Check Qdrant
    try:
        client = get_qdrant_client()
        client.get_collections()
        status_report["qdrant"] = "connected"
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        status_report["qdrant"] = "disconnected"
        
    # Check SQLite
    try:
        execute_query("SELECT 1")
        status_report["sql_db"] = "connected"
    except Exception as e:
        logger.warning(f"SQLite health check failed: {e}")
        status_report["sql_db"] = "disconnected"
        
    # Check Langfuse configuration
    has_keys = bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)
    status_report["langfuse"] = "connected" if has_keys else "enabled (local trace)"
        
    return status_report
