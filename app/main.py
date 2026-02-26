import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
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
from app.stats.football_data_client import FootballDataClient
from app.utils.deduplication import DeduplicationStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Footy Phil — Login</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E⚽%3C/text%3E%3C/svg%3E">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #38003c;
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .card {{
      background: #fff;
      border-radius: 1rem;
      padding: 2.5rem 2rem;
      width: 100%;
      max-width: 360px;
      text-align: center;
      box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }}
    .logo {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}
    h1 {{ font-size: 1.4rem; color: #38003c; margin-bottom: 0.25rem; }}
    p {{ font-size: 0.85rem; color: #888; margin-bottom: 1.75rem; }}
    input {{
      width: 100%;
      border: 1.5px solid #ddd;
      border-radius: 0.6rem;
      padding: 0.7rem 1rem;
      font-size: 1rem;
      outline: none;
      margin-bottom: 1rem;
      transition: border-color 0.15s;
    }}
    input:focus {{ border-color: #38003c; }}
    button {{
      width: 100%;
      background: #38003c;
      color: #fff;
      border: none;
      border-radius: 0.6rem;
      padding: 0.75rem;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }}
    button:hover {{ background: #560060; }}
    .error {{
      background: #fdecea;
      color: #b71c1c;
      border-radius: 0.5rem;
      padding: 0.5rem 0.75rem;
      font-size: 0.85rem;
      margin-bottom: 1rem;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">⚽</div>
    <h1>Footy Phil</h1>
    <p>Your Premier League insider</p>
    {error}
    <form method="post" action="/login">
      <input type="password" name="password" placeholder="Enter password" autofocus autocomplete="current-password">
      <button type="submit">Let me in</button>
    </form>
  </div>
</body>
</html>"""


def _auth_token() -> str:
    return hashlib.sha256(f"phil:{settings.app_password}".encode()).hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Starting Footy Phil...")

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
    stats_client = (
        FootballDataClient(
            settings.football_data_api_key,
            settings.stats_cache_ttl_seconds,
        )
        if settings.football_data_api_key
        else None
    )
    if stats_client:
        logger.info("Live stats enabled (football-data.org, cache TTL %ds)", settings.stats_cache_ttl_seconds)
    chat_engine = ChatEngine(retriever=retriever, llm_client=llm_client, stats_client=stats_client)

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

    logger.info("Footy Phil is ready. Ingestion scheduled every %d minutes.", settings.ingest_interval_minutes)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down...")
    scheduler.shutdown(wait=False)
    await llm_client.close()
    if stats_client:
        await stats_client.close()
    weaviate_manager.close()
    logger.info("Shutdown complete.")


app = FastAPI(title="Footy Phil", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not settings.app_password:
        return await call_next(request)
    # Always allow health check and login routes
    if request.url.path in ("/health", "/login"):
        return await call_next(request)
    if request.cookies.get("auth") == _auth_token():
        return await call_next(request)
    return RedirectResponse("/login")


@app.get("/login", include_in_schema=False)
def login_page():
    return HTMLResponse(_LOGIN_HTML.format(error=""))


@app.post("/login", include_in_schema=False)
async def login_submit(password: str = Form(...)):
    if password == settings.app_password:
        response = RedirectResponse("/", status_code=302)
        response.set_cookie("auth", _auth_token(), httponly=True, samesite="lax")
        return response
    return HTMLResponse(
        _LOGIN_HTML.format(error='<div class="error">Wrong password, mate. Try again.</div>'),
        status_code=401,
    )


app.include_router(router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse("app/static/index.html")
