"""Verifier agent: checks draft answer against source chunks for unsupported claims."""

from __future__ import annotations

import os

import anthropic

from agents.state import AgentState

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


_SYSTEM = (
    "You are a fact-verification assistant. "
    "Given a draft answer and the source passages it was derived from, "
    "determine whether every claim in the draft is directly supported by the sources.\n\n"
    "Reply with exactly one word on the first line: PASS or FAIL.\n"
    "On the second line, briefly explain why (one sentence)."
)

MAX_RETRIES = 2


def verifier_node(state: AgentState) -> dict[str, bool | int]:
    draft = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)

    if not draft or not chunks:
        return {"verified": False, "retry_count": retry_count + 1}

    context = "\n\n---\n\n".join(
        f"[Page {c.page} | {c.section}]\n{c.text}" for c in chunks
    )

    client = _get_client()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=128,
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
                    f"Draft answer:\n{draft}\n\n"
                    "Does the draft contain only claims supported by the sources?"
                ),
            }
        ],
    )

    verdict_text = response.content[0].text.strip()
    passed = verdict_text.upper().startswith("PASS")
    return {"verified": passed, "retry_count": retry_count + (0 if passed else 1)}
