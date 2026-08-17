"""Extraction agent: used for table-heavy or structured-field queries."""

from __future__ import annotations

import os

import anthropic

from agents.state import AgentState
from shared.types import RetrievalResult
from retrieval.hybrid_search import retrieve
from retrieval.rerank import rerank

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


_SYSTEM = (
    "You are a structured-data extraction assistant for financial and legal documents.\n"
    "Given source passages and a question, extract the precise value or values requested.\n"
    "Return the extracted data as a short, direct answer. "
    "If the data isn't present, say 'Not found in provided context.'"
)


def extraction_node(state: AgentState) -> dict[str, list[RetrievalResult] | str]:
    question = state.get("question", "")
    candidates = retrieve(question, top_k=20)
    chunks = rerank(question, candidates, top_k=8)

    context = "\n\n---\n\n".join(
        f"[Page {c.page} | {c.section}]\n{c.text}" for c in chunks
    )

    client = _get_client()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Source passages:\n\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Extract the answer precisely."
                ),
            }
        ],
    )

    draft = response.content[0].text.strip()
    return {"retrieved_chunks": chunks, "draft_answer": draft}
