"""LangGraph StateGraph wiring all five agents."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.state import AgentState
from agents.router import router_node
from agents.retrieval_agent import retrieval_node
from agents.extraction_agent import extraction_node
from agents.verifier_agent import verifier_node
from agents.synthesis_agent import synthesis_node
from shared.types import QueryResponse

MAX_RETRIES = 2


def _route_after_router(state: AgentState) -> str:
    return state.get("routing_decision", "retrieval")


def _route_after_verifier(state: AgentState) -> str:
    if state.get("verified", False):
        return "synthesis"
    retry = state.get("retry_count", 0)
    if retry >= MAX_RETRIES:
        # Give up retrying and synthesize best-effort answer
        return "synthesis"
    return "retrieval"


def build_graph() -> CompiledStateGraph:
    graph: StateGraph[AgentState] = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("synthesis", synthesis_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"retrieval": "retrieval", "extraction": "extraction"},
    )

    graph.add_edge("retrieval", "verifier")
    graph.add_edge("extraction", "verifier")

    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"synthesis": "synthesis", "retrieval": "retrieval"},
    )

    graph.add_edge("synthesis", END)

    return graph.compile()


def answer(question: str) -> QueryResponse:
    app = build_graph()
    initial: AgentState = {
        "question": question,
        "routing_decision": "retrieval",
        "retrieved_chunks": [],
        "draft_answer": "",
        "verified": False,
        "retry_count": 0,
        "final_answer": "",
        "citations": [],
        "confidence": 0.0,
        "error": None,
    }
    result: AgentState = app.invoke(initial)
    return QueryResponse(
        answer=result.get("final_answer", ""),
        citations=list(result.get("citations", [])),
        confidence=float(result.get("confidence", 0.0)),
        verified=bool(result.get("verified", False)),
    )
