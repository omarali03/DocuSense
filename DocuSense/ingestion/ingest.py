"""Main ingestion pipeline. Run directly: python -m ingestion.ingest <folder>"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from shared.types import ParsedChunk, ParsedDocument
from ingestion.parse import parse_document
from ingestion.ocr import ocr_pdf_pages

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer as ST

        _model = ST(EMBEDDING_MODEL)
    return _model


def get_engine() -> Engine:
    return create_engine(os.environ["DATABASE_URL"])


def setup_db(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id      VARCHAR(16) PRIMARY KEY,
                    filename    VARCHAR(512),
                    doc_type    VARCHAR(32),
                    ingested_at TIMESTAMP DEFAULT NOW()
                )
            """)
        )
        conn.execute(
            text(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id     VARCHAR(24) PRIMARY KEY,
                    doc_id       VARCHAR(16) REFERENCES documents(doc_id) ON DELETE CASCADE,
                    text         TEXT,
                    page         INTEGER,
                    section      VARCHAR(256),
                    doc_type     VARCHAR(32),
                    element_type VARCHAR(64),
                    embedding    VECTOR({EMBEDDING_DIM})
                )
            """)
        )
        conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
        )


def _is_ingested(engine: Engine, doc_id: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM documents WHERE doc_id = :d"), {"d": doc_id}
        ).fetchone()
        return row is not None


def _embed(chunks: list[ParsedChunk]) -> list[tuple[ParsedChunk, list[float]]]:
    model = _get_model()
    texts = [c.text for c in chunks]
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return [(c, v.tolist()) for c, v in zip(chunks, vecs)]


def _fmt_vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def _store(engine: Engine, doc: ParsedDocument, pairs: list[tuple[ParsedChunk, list[float]]]) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO documents (doc_id, filename, doc_type) "
                "VALUES (:doc_id, :filename, :doc_type)"
            ),
            {"doc_id": doc.doc_id, "filename": doc.filename, "doc_type": doc.doc_type},
        )
        rows = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "page": c.page,
                "section": c.section,
                "doc_type": c.doc_type,
                "element_type": c.element_type,
                "embedding": _fmt_vec(v),
            }
            for c, v in pairs
        ]
        conn.execute(
            text("""
                INSERT INTO chunks
                    (chunk_id, doc_id, text, page, section, doc_type, element_type, embedding)
                VALUES
                    (:chunk_id, :doc_id, :text, :page, :section, :doc_type, :element_type,
                     CAST(:embedding AS vector))
            """),
            rows,
        )


def ingest_file(path: Path, engine: Engine) -> str | None:
    doc = parse_document(path)

    if _is_ingested(engine, doc.doc_id):
        print(f"skip {path.name} ({doc.doc_id} already ingested)")
        return None

    total_text = sum(len(c.text) for c in doc.chunks)
    if total_text < 100 and path.suffix.lower() == ".pdf":
        print(f"low text yield ({total_text} chars), running OCR on {path.name}")
        ocr_pages = ocr_pdf_pages(path)
        doc.chunks = [
            ParsedChunk(
                chunk_id=f"{doc.doc_id}_{r.page:04d}",
                doc_id=doc.doc_id,
                text=r.text,
                page=r.page,
                section="ocr",
                doc_type=doc.doc_type,
                element_type="OCR",
            )
            for r in ocr_pages
            if r.text.strip()
        ]

    if not doc.chunks:
        print(f"no content from {path.name}")
        return None

    print(f"embedding {len(doc.chunks)} chunks from {path.name}…")
    pairs = _embed(doc.chunks)
    _store(engine, doc, pairs)
    print(f"ingested {path.name} → {doc.doc_id} ({len(doc.chunks)} chunks)")
    return doc.doc_id


def ingest_folder(folder: Path) -> None:
    engine = get_engine()
    setup_db(engine)

    exts = {".pdf", ".docx", ".txt", ".html"}
    files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in exts)
    print(f"found {len(files)} files")

    for path in files:
        try:
            ingest_file(path, engine)
        except Exception as exc:
            print(f"error on {path.name}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    folder_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    ingest_folder(folder_arg)
