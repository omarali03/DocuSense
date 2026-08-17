from __future__ import annotations

import os

import anthropic

from agents.state import AgentState
from shared.types import RoutingDecision

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


_SYSTEM = (
    "You classify document queries into one of two routing targets.\n\n"
    "Reply with ONLY the single word 'extraction' or 'retrieval'.\n\n"
    "Use 'extraction' when the question requires pulling structured data from "
    "a table, chart, or form field (e.g. specific dollar amounts, dates, clause "
    "numbers, or enumerated values that live in a table).\n\n"
    "Use 'retrieval' for all other questions that need narrative context."
)


def router_node(state: AgentState) -> dict[str, RoutingDecision]:
    client = _get_client()
    question = state.get("question", "")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8,
        system=[
            {
                "type": "text",
                "text": _SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": question}],
    )

    raw = response.content[0].text.strip().lower()
    decision: RoutingDecision = "extraction" if "extraction" in raw else "retrieval"
    return {"routing_decision": decision}
