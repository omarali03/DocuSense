"""FastAPI gateway with auth, rate limiting, and tracing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, UploadFile, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

load_dotenv()

# ── Auth ────────────────────────────────────────────────────────────────────

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
_SECRET = os.environ.get("API_SECRET_KEY", "")


def verify_api_key(key: Annotated[str, Security(API_KEY_HEADER)]) -> str:
    if not _SECRET or key != _SECRET:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid API key")
    return key


# ── Rate limiting ────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="DocuSense", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    doc_type: str | None = None


class CitationOut(BaseModel):
    doc_id: str
    chunk_id: str
    page: int
    section: str
    text_excerpt: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    confidence: float
    verified: bool


class IngestResponse(BaseModel):
    doc_id: str | None
    filename: str
    status: str


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def query(request: Request, body: QueryRequest) -> QueryResponse:
    from agents.graph import answer

    result = answer(body.question)
    return QueryResponse(
        answer=result["answer"],
        citations=[CitationOut(**c) for c in result["citations"]],
        confidence=result["confidence"],
        verified=result["verified"],
    )


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def ingest(request: Request, file: UploadFile) -> IngestResponse:
    from ingestion.ingest import get_engine, ingest_file, setup_db

    if file.filename is None:
        raise HTTPException(status_code=400, detail="no filename provided")

    tmp_path = Path("/tmp") / file.filename
    content = await file.read()
    tmp_path.write_bytes(content)

    try:
        engine = get_engine()
        setup_db(engine)
        doc_id = ingest_file(tmp_path, engine)
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        doc_id=doc_id,
        filename=file.filename,
        status="ingested" if doc_id else "skipped",
    )
