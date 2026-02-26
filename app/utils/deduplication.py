import logging
from typing import TYPE_CHECKING

import weaviate.util

from app.config import settings
from app.db.weaviate_client import weaviate_manager

if TYPE_CHECKING:
    from app.ingestion.rss_fetcher import Article

logger = logging.getLogger(__name__)


class DeduplicationStore:
    """
    Two-layer deduplication:
    L1 — in-memory set of UUID5s seen this process lifetime (fast, zero round-trips)
    L2 — Weaviate object existence check (cross-restart safety via deterministic UUID5)

    Because Weaviate uses our deterministic UUID5 as the object ID, inserting a
    duplicate simply overwrites the existing object. The L1 cache lets us skip
    even the upsert call for articles we know are already stored.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def warm_cache(self) -> None:
        """Load all existing object UUIDs from Weaviate into the in-memory set."""
        try:
            collection = weaviate_manager.client.collections.get(settings.collection_name)
            count = 0
            for item in collection.iterator(include_vector=False):
                self._seen.add(str(item.uuid))
                count += 1
            logger.info("Dedup cache warmed with %d existing UUIDs", count)
        except Exception as exc:
            logger.warning("Could not warm dedup cache: %s", exc)

    def filter_new(self, articles: list["Article"]) -> list["Article"]:
        """Return only articles whose UUID5 is not already known."""
        new: list["Article"] = []
        for article in articles:
            uuid = str(weaviate.util.generate_uuid5(article.content_hash))
            if uuid not in self._seen:
                new.append(article)
                self._seen.add(uuid)
        return new
