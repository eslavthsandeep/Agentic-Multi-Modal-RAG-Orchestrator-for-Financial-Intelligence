from __future__ import annotations

import logging
import sqlite3
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

from pathlib import Path

def get_connection() -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
    else:
        db_path = db_url

    path_obj = Path(db_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
        
    conn = sqlite3.connect(str(path_obj))
    conn.row_factory = sqlite3.Row
    return conn

def init_tables() -> None:
    """Initialize necessary database tables."""
    schema = get_table_schema()
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(schema)
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        raise

def execute_query(sql: str, params: tuple = ()) -> dict[str, list[Any]]:
    """Execute a query and return columns and rows."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            
            if cursor.description is None:
                conn.commit()
                return {"columns": [], "rows": []}
                
            columns = [desc[0] for desc in cursor.description]
            rows = [dict(row) for row in cursor.fetchall()]
            
            return {"columns": columns, "rows": rows}
    except Exception as e:
        logger.error(f"Failed to execute query '{sql}': {e}")
        raise

def get_table_schema() -> str:
    """Return the CREATE TABLE statement for context."""
    return """
    CREATE TABLE IF NOT EXISTS stock_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        open_price REAL,
        close_price REAL,
        high_price REAL,
        low_price REAL,
        volume INTEGER
    );
    """
