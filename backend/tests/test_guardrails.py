"""
Tests for NeMo Guardrails integration.
Verifies graceful degradation when config is missing and that the
guard interface returns the expected dict structure.
"""

import os
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


class TestGuardrails:
    """Verify guardrails check_input/check_output behavior and fallback."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_when_config_missing(self):
        """If NeMo config fails to load, guards should pass-through instead of crashing."""
        with patch("app.guardrails.guard._get_rails", return_value=None):
            from app.guardrails.guard import check_input, check_output

            input_result = await check_input("How do I cook pasta?")
            assert input_result["passed"] is True

            output_result = await check_output("Some answer", "Some context")
            assert output_result["passed"] is True

    @pytest.mark.asyncio
    async def test_check_input_returns_expected_shape(self):
        """Verify the return dict always has 'passed' and 'reason' keys."""
        with patch("app.guardrails.guard._get_rails", return_value=None):
            from app.guardrails.guard import check_input

            result = await check_input("What was Q3 revenue?")
            assert "passed" in result
            assert "reason" in result
            assert isinstance(result["passed"], bool)

    @pytest.mark.asyncio
    async def test_check_output_returns_expected_shape(self):
        with patch("app.guardrails.guard._get_rails", return_value=None):
            from app.guardrails.guard import check_output

            result = await check_output("Revenue was $10B", "Revenue was $10B in FY2024")
            assert "passed" in result
            assert "reason" in result

    @pytest.mark.asyncio
    async def test_check_input_blocks_off_topic_when_rails_active(self):
        """When NeMo is loaded, off-topic questions should be caught."""
        mock_rails = AsyncMock()
        # Simulate NeMo returning a refusal
        mock_rails.generate_async.return_value = "I cannot answer questions about cooking."

        with patch("app.guardrails.guard._get_rails", return_value=mock_rails):
            from app.guardrails.guard import check_input

            result = await check_input("How do I bake a cake?")
            assert result["passed"] is False
