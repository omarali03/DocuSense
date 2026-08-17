from __future__ import annotations

from typing import TypedDict

from shared.types import Citation, RetrievalResult, RoutingDecision


class AgentState(TypedDict, total=False):
    question: str
    routing_decision: RoutingDecision
    retrieved_chunks: list[RetrievalResult]
    draft_answer: str
    verified: bool
    retry_count: int
    final_answer: str
    citations: list[Citation]
    confidence: float
    error: str | None
