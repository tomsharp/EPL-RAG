from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SourceDoc(BaseModel):
    title: str
    url: str
    published: Optional[str] = None
    source: str
    score: float


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class HistoryTurn(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceDoc]
    retrieved_doc_count: int


class IngestRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="Re-embed articles even if already stored (full re-index)",
    )


class IngestResponse(BaseModel):
    status: str
    articles_fetched: int
    articles_embedded: int
    articles_skipped: int
    duration_seconds: float


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class HealthResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    weaviate_connected: bool
    collection_exists: bool
    total_objects: int
    last_ingest_time: Optional[datetime] = None
    next_ingest_time: Optional[datetime] = None
    scheduler_running: bool
