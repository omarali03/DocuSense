from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

DocType = Literal["10-K", "10-Q", "contract", "form", "unknown"]
RoutingDecision = Literal["retrieval", "extraction"]


@dataclass
class ParsedChunk:
    chunk_id: str
    doc_id: str
    text: str
    page: int
    section: str
    doc_type: DocType
    element_type: str


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    doc_type: DocType
    chunks: list[ParsedChunk] = field(default_factory=list)


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    text: str
    page: int
    section: str
    doc_type: str
    score: float


class Citation(TypedDict):
    doc_id: str
    chunk_id: str
    page: int
    section: str
    text_excerpt: str


class QueryResponse(TypedDict):
    answer: str
    citations: list[Citation]
    confidence: float
    verified: bool
