import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import weaviate.util

from app.config import settings
from app.db.weaviate_client import weaviate_manager
from app.ingestion.embedder import Embedder
from app.ingestion.rss_fetcher import Article, RSSFetcher

if TYPE_CHECKING:
    from app.utils.deduplication import DeduplicationStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        fetcher: RSSFetcher,
        embedder: Embedder,
        dedup_store: "DeduplicationStore",
    ) -> None:
        self.fetcher = fetcher
        self.embedder = embedder
        self.dedup = dedup_store
        self.last_run: datetime | None = None

    def run(self, force: bool = False) -> dict:
        start = time.perf_counter()
        logger.info("Starting ingestion pipeline (force=%s)", force)

        # 1. Fetch articles from all RSS feeds
        articles = self.fetcher.fetch_all()
        fetched_count = len(articles)

        # 2. Filter out already-seen articles (unless force=True)
        if force:
            new_articles = articles
            skipped_count = 0
        else:
            new_articles = self.dedup.filter_new(articles)
            skipped_count = fetched_count - len(new_articles)

        if not new_articles:
            logger.info(
                "No new articles to embed (fetched=%d, skipped=%d)", fetched_count, skipped_count
            )
            self.last_run = datetime.now(timezone.utc)
            return {
                "articles_fetched": fetched_count,
                "articles_embedded": 0,
                "articles_skipped": skipped_count,
                "duration_seconds": round(time.perf_counter() - start, 2),
            }

        # 3. Build texts to embed: title + summary
        embed_texts = [f"{a.title}. {a.summary}" for a in new_articles]

        # 4. Embed
        vectors = self.embedder.encode(embed_texts)

        # 5. Batch upsert to Weaviate
        collection = weaviate_manager.client.collections.get(settings.collection_name)
        with collection.batch.dynamic() as batch:
            for article, vector in zip(new_articles, vectors):
                uuid = weaviate.util.generate_uuid5(article.content_hash)
                props = self._build_payload(article)
                batch.add_object(
                    properties=props,
                    vector=vector.tolist(),
                    uuid=uuid,
                )

        embedded_count = len(new_articles)
        self.last_run = datetime.now(timezone.utc)

        logger.info(
            "Ingestion complete: fetched=%d embedded=%d skipped=%d",
            fetched_count,
            embedded_count,
            skipped_count,
        )

        return {
            "articles_fetched": fetched_count,
            "articles_embedded": embedded_count,
            "articles_skipped": skipped_count,
            "duration_seconds": round(time.perf_counter() - start, 2),
        }

    def _build_payload(self, article: Article) -> dict:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        published_iso: str | None = None
        if article.published is not None:
            # Weaviate DATE requires RFC3339 — isoformat() produces +00:00 for UTC,
            # which is valid. For naive datetimes, assume UTC and append Z.
            if article.published.tzinfo is None:
                published_iso = article.published.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                published_iso = article.published.isoformat()

        return {
            "title": article.title,
            "url": article.url,
            "summary": article.summary,
            "source": article.source,
            "published": published_iso,
            "content_hash": article.content_hash,
            "ingested_at": now_iso,
        }
