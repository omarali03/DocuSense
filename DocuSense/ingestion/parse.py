from __future__ import annotations

import hashlib
from pathlib import Path

from shared.types import DocType, ParsedChunk, ParsedDocument


def detect_doc_type(filename: str) -> DocType:
    name = filename.lower()
    if "10-k" in name or "annual" in name:
        return "10-K"
    if "10-q" in name or "quarterly" in name:
        return "10-Q"
    if "contract" in name or "agreement" in name:
        return "contract"
    if "form" in name or "funsd" in name:
        return "form"
    return "unknown"


def parse_document(path: Path, doc_type: DocType | None = None) -> ParsedDocument:
    raw = path.read_bytes()
    doc_id = hashlib.sha256(raw).hexdigest()[:16]
    resolved_type = doc_type or detect_doc_type(path.name)

    # Lazy import so the module loads even if unstructured isn't installed yet
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import Title

    try:
        elements = partition(filename=str(path), strategy="hi_res")
    except Exception:
        elements = partition(filename=str(path), strategy="fast")

    chunks: list[ParsedChunk] = []
    current_section = "preamble"
    chunk_index = 0

    for el in elements:
        if isinstance(el, Title):
            current_section = (el.text or "").strip()[:128]
            continue

        text = (el.text or "").strip()
        if not text:
            continue

        page_num = 0
        if hasattr(el, "metadata") and el.metadata is not None:
            pn = getattr(el.metadata, "page_number", None)
            if isinstance(pn, int):
                page_num = pn

        chunks.append(
            ParsedChunk(
                chunk_id=f"{doc_id}_{chunk_index:04d}",
                doc_id=doc_id,
                text=text,
                page=page_num,
                section=current_section,
                doc_type=resolved_type,
                element_type=type(el).__name__,
            )
        )
        chunk_index += 1

    return ParsedDocument(
        doc_id=doc_id,
        filename=path.name,
        doc_type=resolved_type,
        chunks=chunks,
    )
