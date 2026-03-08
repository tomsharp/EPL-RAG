import logging
import uuid

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    HealthResponse,
    HistoryTurn,
    IngestRequest,
    IngestResponse,
    SourceDoc,
    StatusResponse,
)

_SESSION_COOKIE = "sid"
_SESSION_MAX_AGE = 90 * 24 * 3600  # 90 days
from app.config import settings
from app.db.weaviate_client import weaviate_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, request: Request, response: Response) -> ChatResponse:
    session_id = request.cookies.get(_SESSION_COOKIE) or str(uuid.uuid4())
    response.set_cookie(
        key=_SESSION_COOKIE,
        value=session_id,
        max_age=_SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
    )

    chat_engine = request.app.state.chat_engine

    try:
        result = await chat_engine.chat(session_id, body.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception:
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
        session_id=session_id,
        answer=result["answer"],
        sources=sources,
        retrieved_doc_count=result["retrieved_doc_count"],
    )


@router.get("/history", response_model=list[HistoryTurn])
async def history_endpoint(request: Request) -> list[HistoryTurn]:
    session_id = request.cookies.get(_SESSION_COOKIE)
    if not session_id:
        return []
    conv_repo = getattr(request.app.state, "conv_repo", None)
    if not conv_repo:
        return []
    turns = await conv_repo.get_history(session_id, max_turns=100)
    return [HistoryTurn(role=t["role"], content=t["content"]) for t in turns]


@router.post("/chat/stream")
async def chat_stream_endpoint(body: ChatRequest, request: Request, response: Response) -> StreamingResponse:
    session_id = request.cookies.get(_SESSION_COOKIE) or str(uuid.uuid4())
    chat_engine = request.app.state.chat_engine

    async def event_generator():
        try:
            async for chunk in chat_engine.chat_stream(session_id, body.message):
                yield chunk
        except Exception:
            logger.exception("Streaming error in /chat/stream")
            yield f'data: {{"type":"error","message":"Internal server error"}}\n\n'

    streaming_response = StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    streaming_response.set_cookie(
        key=_SESSION_COOKIE,
        value=session_id,
        max_age=_SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return streaming_response


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


@router.post("/feedback", status_code=204)
async def feedback_endpoint(body: FeedbackRequest) -> None:
    if not settings.resend_api_key or not settings.feedback_email_to:
        raise HTTPException(status_code=503, detail="Feedback not configured")
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {settings.resend_api_key}"},
                json={
                    "from": settings.feedback_email_from,
                    "to": [settings.feedback_email_to],
                    "subject": "EPL Insider Feedback",
                    "text": body.text,
                },
                timeout=10.0,
            )
            if not res.is_success:
                logger.error("Resend %s: %s", res.status_code, res.text)
                res.raise_for_status()
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=500, detail="Failed to send feedback")
    except Exception:
        logger.exception("Resend request failed")
        raise HTTPException(status_code=500, detail="Failed to send feedback")


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
