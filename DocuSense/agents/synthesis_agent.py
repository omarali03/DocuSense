"""Synthesis agent: produces the final answer with inline citations."""

from __future__ import annotations

import os

import anthropic

from agents.state import AgentState
from shared.types import Citation

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


_SYSTEM = (
    "You are a document intelligence assistant. "
    "Answer the user's question using ONLY the provided source passages. "
    "After each claim, add an inline citation like [doc_id, p.N]. "
    "End with a one-line confidence note: Confidence: HIGH / MEDIUM / LOW."
)


def synthesis_node(state: AgentState) -> dict[str, str | list[Citation] | float]:
    question = state.get("question", "")
    chunks = state.get("retrieved_chunks", [])
    draft = state.get("draft_answer", "")

    context_parts: list[str] = []
    for c in chunks:
        context_parts.append(f"[{c.doc_id}, p.{c.page} | {c.section}]\n{c.text}")
    context = "\n\n---\n\n".join(context_parts)

    # If extraction already produced a draft, use it as a seed
    seed = f"\nPrevious extraction draft: {draft}\n" if draft else ""

    client = _get_client()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
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
                    f"Source passages:\n\n{context}\n{seed}\n"
                    f"Question: {question}\n\n"
                    "Provide a complete, cited answer."
                ),
            }
        ],
    )

    answer_text = response.content[0].text.strip()

    confidence_line = [l for l in answer_text.splitlines() if l.startswith("Confidence:")]
    raw_conf = confidence_line[-1] if confidence_line else "Confidence: MEDIUM"
    conf_map = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
    confidence = conf_map.get(raw_conf.split(":")[-1].strip().upper(), 0.6)

    citations: list[Citation] = [
        Citation(
            doc_id=c.doc_id,
            chunk_id=c.chunk_id,
            page=c.page,
            section=c.section,
            text_excerpt=c.text[:200],
        )
        for c in chunks
    ]

    return {
        "final_answer": answer_text,
        "citations": citations,
        "confidence": confidence,
    }
