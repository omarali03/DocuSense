from __future__ import annotations

from agents.state import AgentState
from shared.types import RetrievalResult
from retrieval.hybrid_search import retrieve
from retrieval.rerank import rerank


def retrieval_node(state: AgentState) -> dict[str, list[RetrievalResult]]:
    question = state.get("question", "")
    doc_type = state.get("routing_decision")
    retry = state.get("retry_count", 0)

    # Broaden top_k on retry to give verifier more to work with
    top_k = 10 + retry * 5

    candidates = retrieve(question, top_k=top_k * 2)
    ranked = rerank(question, candidates, top_k=top_k)
    return {"retrieved_chunks": ranked}
