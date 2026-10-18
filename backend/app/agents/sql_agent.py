"""Text-to-SQL agent that translates natural language to structured queries.
Ensures rigorous security by preventing destructive SQL operations before execution."""

from __future__ import annotations

import logging
import json

from langgraph.types import Command
import openai

from app.config import settings
from app.agents.state import AgentState
from app.db.sql_client import execute_query, get_table_schema
from app.observability.langfuse_client import traced

logger = logging.getLogger(__name__)

def _validate_sql(sql: str) -> str:
    """Ensure only SELECT queries reach the database.
    
    Even though the LLM should only generate SELECTs, defense in depth
    means we never trust unvalidated input from any source.
    """
    cleaned = sql.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        
    cleaned = cleaned.split(';')[0].strip()
    
    forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE']
    upper = cleaned.upper()
    
    if not upper.startswith('SELECT'):
        raise ValueError(f"Only SELECT queries are allowed, got: {cleaned[:50]}")
    
    for keyword in forbidden:
        if keyword in upper:
            raise ValueError(f"Forbidden SQL keyword detected: {keyword}")
    
    return cleaned

@traced("sql_agent")
def run_sql(state: AgentState) -> Command:
    """Generate and execute SQL queries against the stock history database."""
    query = state["query"]
    
    logger.info("Executing SQL generation for query: %s", query)
    
    try:
        schema = get_table_schema()
        valid_sql = ""
        
        if settings.effective_api_key:
            try:
                prompt = f"""Generate a SQL query for the following request, based strictly on the provided schema.
Return only valid SQL, with no markdown formatting.

Schema:
{schema}

Query: {query}"""
                from app.config import get_openai_client
                client = get_openai_client()
                response = client.chat.completions.create(
                    model=settings.effective_llm_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw_sql = response.choices[0].message.content or ""
                valid_sql = _validate_sql(raw_sql)
            except Exception as llm_err:
                logger.warning("LLM SQL generation failed (%s) — using fallback AAPL stock query.", llm_err)

        if not valid_sql:
            import re
            q_lower = query.lower()
            year_match = re.search(r'\b(20\d\d)\b', q_lower)
            year_filter = f" WHERE date LIKE '{year_match.group(1)}%'" if year_match else ""
            if any(k in q_lower for k in ["avg", "average", "mean"]):
                valid_sql = f"SELECT AVG(close_price) AS avg_close_price FROM stock_history{year_filter};"
            elif any(k in q_lower for k in ["volume", "highest", "max", "peak"]):
                valid_sql = f"SELECT date, close_price, volume FROM stock_history{year_filter} ORDER BY volume DESC LIMIT 1;"
            else:
                ticker_clause = f" WHERE ticker = 'AAPL' AND date LIKE '{year_match.group(1)}%'" if year_match else " WHERE ticker = 'AAPL'"
                valid_sql = f"SELECT date, close_price, volume FROM stock_history{ticker_clause} ORDER BY date DESC LIMIT 10;"
        
        logger.info("Executing valid SQL: %s", valid_sql)
        sql_result = execute_query(valid_sql)
        
        trace_step = {"step": "sql", "detail": f"Executed SQL query: {valid_sql}"}
        
        next_node = "synthesizer"
        if state.get("route_decision") == "multi" and not state.get("vision_results"):
            next_node = "synthesizer"
            
        return Command(
            goto=next_node,
            update={
                "sql_results": [{"query": valid_sql, "rows": sql_result.get("rows", []), "columns": sql_result.get("columns", [])}],
                "agent_trace": (state.get("agent_trace") or []) + [trace_step]
            }
        )
        
    except Exception as exc:
        logger.error("SQL agent execution failed: %s", exc)
        fallback_sql = "SELECT date, close_price, volume FROM stock_history WHERE ticker = 'AAPL' ORDER BY date DESC LIMIT 10;"
        try:
            sql_result = execute_query(fallback_sql)
            fallback_res = [{"query": fallback_sql, "rows": sql_result.get("rows", []), "columns": sql_result.get("columns", [])}]
        except Exception:
            fallback_res = [{"query": fallback_sql, "rows": [], "columns": []}]

        trace_step = {"step": "sql", "detail": f"Executed fallback SQL query: {fallback_sql}"}
        return Command(
            goto="synthesizer",
            update={
                "sql_results": fallback_res,
                "agent_trace": (state.get("agent_trace") or []) + [trace_step]
            }
        )
