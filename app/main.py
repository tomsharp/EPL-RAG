import asyncio
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.db.weaviate_client import weaviate_manager
from app.ingestion.embedder import Embedder
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.rss_fetcher import RSSFetcher
from app.rag.chat_engine import ChatEngine
from app.rag.llm_client import LLMClient
from app.rag.retriever import Retriever
from app.utils.deduplication import DeduplicationStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Starting EPL RAG Chatbot...")

    # 1. Connect to Weaviate and ensure collection + schema exist
    weaviate_manager.connect()
    weaviate_manager.ensure_collection()

    # 2. Initialize shared components
    embedder = Embedder(settings.embedding_model)
    dedup_store = DeduplicationStore()

    # 3. Warm the deduplication cache from existing Weaviate data
    dedup_store.warm_cache()

    # 4. Build the ingestion pipeline
    fetcher = RSSFetcher()
    pipeline = IngestionPipeline(fetcher=fetcher, embedder=embedder, dedup_store=dedup_store)

    # 5. Build the RAG chain
    retriever = Retriever(embedder=embedder)
    llm_client = LLMClient()
    chat_engine = ChatEngine(retriever=retriever, llm_client=llm_client)

    # 6. Attach to app.state for route handlers
    app.state.pipeline = pipeline
    app.state.chat_engine = chat_engine

    # 7. Seed the database on startup (run in thread pool so the event loop stays free)
    logger.info("Seeding database with latest EPL news...")
    loop = asyncio.get_event_loop()
    seed_stats = await loop.run_in_executor(None, pipeline.run)
    logger.info(
        "Seed complete: fetched=%d embedded=%d skipped=%d (%.1fs)",
        seed_stats["articles_fetched"],
        seed_stats["articles_embedded"],
        seed_stats["articles_skipped"],
        seed_stats["duration_seconds"],
    )

    # 8. Schedule periodic ingestion
    scheduler = AsyncIOScheduler(
        timezone="UTC",
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 60,
        },
    )
    scheduler.add_job(
        func=pipeline.run,
        trigger=IntervalTrigger(minutes=settings.ingest_interval_minutes),
        id="periodic_ingest",
        replace_existing=True,
        name="EPL RSS Ingest",
    )
    scheduler.start()
    app.state.scheduler = scheduler

    logger.info(
        "EPL RAG Chatbot ready. Ingestion scheduled every %d minutes.",
        settings.ingest_interval_minutes,
    )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down...")
    scheduler.shutdown(wait=False)
    await llm_client.close()
    weaviate_manager.close()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="EPL RAG Chatbot",
    description=(
        "Retrieval-Augmented Generation chatbot for English Premier League news. "
        "Continuously ingests EPL news from RSS feeds into Weaviate and answers "
        "questions using HuggingFace Inference API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse("app/static/index.html")
