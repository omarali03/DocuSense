"""Hybrid retrieval: dense cosine (pgvector) + BM25, fused with RRF."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from shared.types import RetrievalResult

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RRF_K = 60


@lru_cache(maxsize=1)
def _get_embed_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer as ST

    return ST(EMBEDDING_MODEL)


def _get_engine() -> Engine:
    return create_engine(os.environ["DATABASE_URL"])


def _embed_query(query: str) -> list[float]:
    model = _get_embed_model()
    vec = model.encode([query], normalize_embeddings=True)[0]
    return [float(x) for x in vec]


def _fmt_vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def _dense_search(query_vec: list[float], top_k: int, doc_type: str | None) -> list[RetrievalResult]:
    engine = _get_engine()
    filter_clause = "AND doc_type = :doc_type" if doc_type else ""
    params: dict[str, object] = {"vec": _fmt_vec(query_vec), "k": top_k}
    if doc_type:
        params["doc_type"] = doc_type

    sql = text(f"""
        SELECT chunk_id, doc_id, text, page, section, doc_type,
               1 - (embedding <=> CAST(:vec AS vector)) AS score
        FROM chunks
        WHERE 1=1 {filter_clause}
        ORDER BY embedding <=> CAST(:vec AS vector)
        LIMIT :k
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [
        RetrievalResult(
            chunk_id=str(r.chunk_id),
            doc_id=str(r.doc_id),
            text=str(r.text),
            page=int(r.page),
            section=str(r.section),
            doc_type=str(r.doc_type),
            score=float(r.score),
        )
        for r in rows
    ]


def _load_bm25_corpus(doc_type: str | None) -> tuple[list[RetrievalResult], BM25Okapi]:
    from rank_bm25 import BM25Okapi

    engine = _get_engine()
    filter_clause = "WHERE doc_type = :doc_type" if doc_type else ""
    params: dict[str, object] = {"doc_type": doc_type} if doc_type else {}

    sql = text(f"SELECT chunk_id, doc_id, text, page, section, doc_type FROM chunks {filter_clause}")
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    corpus = [
        RetrievalResult(
            chunk_id=str(r.chunk_id),
            doc_id=str(r.doc_id),
            text=str(r.text),
            page=int(r.page),
            section=str(r.section),
            doc_type=str(r.doc_type),
            score=0.0,
        )
        for r in rows
    ]
    tokenized = [c.text.lower().split() for c in corpus]
    return corpus, BM25Okapi(tokenized)


def _bm25_search(query: str, top_k: int, doc_type: str | None) -> list[RetrievalResult]:
    corpus, bm25 = _load_bm25_corpus(doc_type)
    if not corpus:
        return []

    scores: list[float] = bm25.get_scores(query.lower().split()).tolist()
    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for chunk, score in ranked:
        results.append(
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
                doc_type=chunk.doc_type,
                score=score,
            )
        )
    return results


def _rrf_fuse(
    dense: list[RetrievalResult],
    sparse: list[RetrievalResult],
    top_k: int,
) -> list[RetrievalResult]:
    scores: dict[str, float] = {}
    index: dict[str, RetrievalResult] = {}

    for rank, r in enumerate(dense):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        index[r.chunk_id] = r

    for rank, r in enumerate(sparse):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        index.setdefault(r.chunk_id, r)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    fused = []
    for chunk_id, score in ranked:
        r = index[chunk_id]
        fused.append(
            RetrievalResult(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                text=r.text,
                page=r.page,
                section=r.section,
                doc_type=r.doc_type,
                score=score,
            )
        )
    return fused


def retrieve(
    query: str,
    top_k: int = 10,
    doc_type: str | None = None,
) -> list[RetrievalResult]:
    query_vec = _embed_query(query)
    dense = _dense_search(query_vec, top_k * 2, doc_type)
    sparse = _bm25_search(query, top_k * 2, doc_type)
    return _rrf_fuse(dense, sparse, top_k)
