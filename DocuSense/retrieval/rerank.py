"""Cross-encoder reranking on top-k candidates."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from shared.types import RetrievalResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    from sentence_transformers import CrossEncoder as CE

    return CE(RERANKER_MODEL)


def rerank(query: str, candidates: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores: list[float] = reranker.predict(pairs).tolist()

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        RetrievalResult(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            text=r.text,
            page=r.page,
            section=r.section,
            doc_type=r.doc_type,
            score=score,
        )
        for r, score in ranked
    ]
