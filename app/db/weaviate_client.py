import logging

import weaviate
import weaviate.classes as wvc
from weaviate.exceptions import UnexpectedStatusCodeError

from app.config import settings

logger = logging.getLogger(__name__)


class WeaviateManager:
    def __init__(self) -> None:
        self.client: weaviate.WeaviateClient | None = None

    def connect(self) -> None:
        self.client = weaviate.connect_to_local(
            host=settings.weaviate_host,
            port=settings.weaviate_port,
            grpc_port=settings.weaviate_grpc_port,
        )
        logger.info(
            "Connected to Weaviate at %s:%d",
            settings.weaviate_host,
            settings.weaviate_port,
        )

    def close(self) -> None:
        if self.client:
            self.client.close()
            logger.info("Weaviate connection closed")

    def ensure_collection(self) -> None:
        if self.client.collections.exists(settings.collection_name):
            logger.info("Collection '%s' already exists", settings.collection_name)
            return

        self.client.collections.create(
            name=settings.collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE,
            ),
            properties=[
                wvc.config.Property(
                    name="title",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="url",
                    data_type=wvc.config.DataType.TEXT,
                    skip_vectorization=True,
                    index_filterable=True,
                    index_searchable=False,
                ),
                wvc.config.Property(
                    name="summary",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="source",
                    data_type=wvc.config.DataType.TEXT,
                    skip_vectorization=True,
                    index_filterable=True,
                    index_searchable=False,
                ),
                wvc.config.Property(
                    name="published",
                    data_type=wvc.config.DataType.DATE,
                    skip_vectorization=True,
                    index_filterable=True,
                    index_searchable=False,
                ),
                wvc.config.Property(
                    name="content_hash",
                    data_type=wvc.config.DataType.TEXT,
                    skip_vectorization=True,
                    index_filterable=True,
                    index_searchable=False,
                ),
                wvc.config.Property(
                    name="ingested_at",
                    data_type=wvc.config.DataType.DATE,
                    skip_vectorization=True,
                    index_filterable=False,
                    index_searchable=False,
                ),
            ],
        )
        logger.info("Created collection '%s'", settings.collection_name)

    def is_healthy(self) -> bool:
        try:
            return self.client is not None and self.client.is_ready()
        except Exception:
            return False

    def get_total_objects(self) -> int:
        try:
            info = self.client.collections.get(settings.collection_name).aggregate.over_all(
                total_count=True
            )
            return info.total_count or 0
        except Exception:
            return 0


weaviate_manager = WeaviateManager()
