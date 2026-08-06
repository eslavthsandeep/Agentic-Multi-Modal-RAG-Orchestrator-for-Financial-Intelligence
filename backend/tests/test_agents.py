"""
Tests for agent modules — SQL validation guards and supervisor routing logic.
These tests verify security boundaries and correct query dispatch without
requiring real API keys or running services.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.agents.sql_agent import _validate_sql


# ── SQL validation guard tests ──────────────────────────────────────────────

class TestSqlValidation:
    """Verify the defense-in-depth SQL guard rejects anything that isn't a SELECT."""

    @pytest.mark.parametrize("query", [
        "SELECT * FROM stock_history",
        "SELECT AVG(close_price) FROM stock_history WHERE ticker = 'AAPL'",
        "  select count(*) from stock_history  ",
    ])
    def test_allows_valid_selects(self, query):
        result = _validate_sql(query)
        assert result.upper().startswith("SELECT")

    @pytest.mark.parametrize("query,keyword", [
        ("DELETE FROM stock_history WHERE id = 1", "DELETE"),
        ("DROP TABLE stock_history", "DROP"),
        ("INSERT INTO stock_history VALUES (1, 'AAPL', '2024-01-01', 100, 105, 110, 95, 1000000)", "INSERT"),
        ("UPDATE stock_history SET close_price = 0", "UPDATE"),
        ("TRUNCATE TABLE stock_history", "TRUNCATE"),
        ("ALTER TABLE stock_history ADD COLUMN hack TEXT", "ALTER"),
        ("CREATE TABLE evil (id INT)", "CREATE"),
    ])
    def test_blocks_destructive_operations(self, query, keyword):
        with pytest.raises(ValueError, match=f"(?i){keyword}|only select"):
            _validate_sql(query)

    def test_strips_trailing_statements(self):
        """Multi-statement injection: only the first statement (before ';') is kept."""
        result = _validate_sql("SELECT * FROM stock_history; DROP TABLE stock_history")
        assert "DROP" not in result
        assert result == "SELECT * FROM stock_history"

    def test_rejects_non_select_start(self):
        with pytest.raises(ValueError, match="Only SELECT"):
            _validate_sql("EXPLAIN SELECT * FROM stock_history")

    def test_handles_whitespace_and_casing(self):
        result = _validate_sql("   SELECT ticker FROM stock_history   ")
        assert result == "SELECT ticker FROM stock_history"


# ── Supervisor routing tests ────────────────────────────────────────────────

class TestSupervisorRouting:
    """Verify the supervisor dispatches queries to the correct agent based on LLM output."""

    def _make_mock_llm_response(self, route: str, reasoning: str = "test"):
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({"route": route, "reasoning": reasoning})
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    def _build_state(self, query: str) -> dict:
        return {
            "query": query,
            "document_id": "test_doc",
            "chat_history": [],
            "messages": [],
            "route_decision": None,
            "search_results": None,
            "sql_results": None,
            "vision_results": None,
            "self_correction_attempts": 0,
            "final_answer": None,
            "citations": [],
            "agent_trace": [],
            "guardrail_flag": None,
        }

    @patch("app.agents.supervisor.openai")
    def test_routes_stock_question_to_sql(self, mock_openai_module):
        from app.agents.supervisor import supervisor_node

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_llm_response("sql")
        mock_openai_module.OpenAI.return_value = mock_client

        state = self._build_state("What was AAPL's average closing price in 2023?")
        result = supervisor_node(state)
        assert result.goto == "sql_agent"

    @patch("app.agents.supervisor.openai")
    def test_routes_chart_question_to_vision(self, mock_openai_module):
        from app.agents.supervisor import supervisor_node

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_llm_response("vision")
        mock_openai_module.OpenAI.return_value = mock_client

        state = self._build_state("What does the revenue bar chart on page 15 show?")
        result = supervisor_node(state)
        assert result.goto == "vision_agent"

    @patch("app.agents.supervisor.openai")
    def test_routes_text_question_to_search(self, mock_openai_module):
        from app.agents.supervisor import supervisor_node

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_llm_response("search")
        mock_openai_module.OpenAI.return_value = mock_client

        state = self._build_state("Summarize the risk factors section")
        result = supervisor_node(state)
        assert result.goto == "search_agent"
