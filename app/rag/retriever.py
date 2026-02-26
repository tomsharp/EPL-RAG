import logging

import weaviate.classes as wvc

from app.config import settings
from app.db.weaviate_client import weaviate_manager
from app.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)


class SourceDoc:
    def __init__(
        self,
        title: str,
        url: str,
        summary: str,
        published: str | None,
        source: str,
        score: float,
    ) -> None:
        self.title = title
        self.url = url
        self.summary = summary
        self.published = published
        self.source = source
        self.score = score


class Retriever:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def search(self, query: str, top_k: int | None = None) -> list[SourceDoc]:
        k = top_k or settings.max_context_docs
        query_vector = self.embedder.encode_query(query)

        try:
            collection = weaviate_manager.client.collections.get(settings.collection_name)
            results = collection.query.near_vector(
                near_vector=query_vector,
                limit=k,
                return_metadata=wvc.query.MetadataQuery(certainty=True),
                return_properties=["title", "url", "summary", "source", "published"],
            )
        except Exception as exc:
            logger.error("Weaviate search failed: %s", exc)
            return []

        docs: list[SourceDoc] = []
        for obj in results.objects:
            p = obj.properties
            score = obj.metadata.certainty if obj.metadata and obj.metadata.certainty else 0.0
            docs.append(
                SourceDoc(
                    title=p.get("title", ""),
                    url=p.get("url", ""),
                    summary=p.get("summary", ""),
                    published=p.get("published"),
                    source=p.get("source", ""),
                    score=round(score, 4),
                )
            )

        logger.debug("Retrieved %d docs for query: %.60s...", len(docs), query)
        return docs

    def search_with_context(
        self, query: str, top_k: int | None = None
    ) -> tuple[str, list[SourceDoc]]:
        docs = self.search(query, top_k)

        if not docs:
            return "No relevant articles found in the knowledge base.", docs

        parts: list[str] = []
        for doc in docs:
            pub = doc.published or "Unknown date"
            parts.append(
                f"[{doc.source.upper()}] {doc.title} ({pub})\n"
                f"{doc.summary}"
            )

        context = "\n\n".join(parts)
        return context, docs
