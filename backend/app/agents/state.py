"""Defines the core state schema for the LangGraph orchestration.
This state flows through all agents, capturing routing decisions, search results,
and the final synthesized answer for financial document analysis."""

from typing import TypedDict, Literal, Annotated, Optional, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    query: str
    document_id: str
    chat_history: List[dict]
    messages: Annotated[List[BaseMessage], add_messages]
    route_decision: Optional[str]
    search_results: Optional[List[dict]]
    sql_results: Optional[List[dict]]
    vision_results: Optional[List[dict]]
    self_correction_attempts: int
    final_answer: Optional[str]
    citations: List[dict]
    agent_trace: List[dict]
    guardrail_flag: Optional[str]
