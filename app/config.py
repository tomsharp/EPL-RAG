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

    # Live stats — football-data.org (leave blank to disable)
    football_data_api_key: str = ""
    stats_cache_ttl_seconds: int = 600

    # Auth (leave blank to disable password protection)
    app_password: str = ""

    # Conversation persistence (Postgres)
    database_url: str = ""

    # Feedback email — Resend (leave blank to disable)
    resend_api_key: str = ""
    feedback_email_to: str = ""
    feedback_email_from: str = "EPL Insider <feedback@resend.dev>"

    # Observability — Arize Phoenix Cloud (leave blank to disable)
    phoenix_api_key: str = ""
    phoenix_collector_endpoint: str = ""
    phoenix_project: str = "default"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
