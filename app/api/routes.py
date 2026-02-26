import logging
import time

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceDoc,
    StatusResponse,
)
from app.config import settings
from app.db.weaviate_client import weaviate_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, request: Request) -> ChatResponse:
    chat_engine = request.app.state.chat_engine

    try:
        result = await chat_engine.chat(body.session_id, body.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error")

    sources = [
        SourceDoc(
            title=doc.title,
            url=doc.url,
            published=str(doc.published) if doc.published else None,
            source=doc.source,
            score=doc.score,
        )
        for doc in result["sources"]
    ]

    return ChatResponse(
        session_id=body.session_id,
        answer=result["answer"],
        sources=sources,
        retrieved_doc_count=result["retrieved_doc_count"],
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(body: IngestRequest, request: Request) -> IngestResponse:
    pipeline = request.app.state.pipeline

    try:
        stats = pipeline.run(force=body.force)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}")

    return IngestResponse(
        status="ok",
        articles_fetched=stats["articles_fetched"],
        articles_embedded=stats["articles_embedded"],
        articles_skipped=stats["articles_skipped"],
        duration_seconds=stats["duration_seconds"],
    )


@router.get("/health", response_model=HealthResponse)
def health_endpoint() -> HealthResponse:
    if weaviate_manager.is_healthy():
        return HealthResponse(status="healthy")
    return HealthResponse(status="degraded")


@router.get("/status", response_model=StatusResponse)
def status_endpoint(request: Request) -> StatusResponse:
    pipeline = request.app.state.pipeline
    scheduler = request.app.state.scheduler

    connected = weaviate_manager.is_healthy()
    collection_exists = False
    total_objects = 0

    if connected:
        try:
            collection_exists = weaviate_manager.client.collections.exists(
                settings.collection_name
            )
            if collection_exists:
                total_objects = weaviate_manager.get_total_objects()
        except Exception:
            pass

    next_run_time = None
    scheduler_running = False
    if scheduler:
        scheduler_running = scheduler.running
        job = scheduler.get_job("periodic_ingest")
        if job:
            next_run_time = job.next_run_time

    return StatusResponse(
        weaviate_connected=connected,
        collection_exists=collection_exists,
        total_objects=total_objects,
        last_ingest_time=pipeline.last_run,
        next_ingest_time=next_run_time,
        scheduler_running=scheduler_running,
    )
