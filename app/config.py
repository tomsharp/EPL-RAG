from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # Weaviate
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080
    weaviate_grpc_port: int = 50051
    weaviate_secure: bool = False  # set True when connecting over HTTPS (e.g. Fly.io)
    collection_name: str = "EplNews"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Ingestion
    ingest_interval_minutes: int = 30

    # RAG
    max_history_turns: int = 5
    max_context_docs: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
