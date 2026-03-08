"""Conversation persistence — Postgres via asyncpg connection pool."""

import json
import uuid

import asyncpg

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id         TEXT PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id           TEXT PRIMARY KEY,
        session_id   TEXT NOT NULL REFERENCES sessions(id),
        role         TEXT NOT NULL,
        content      TEXT NOT NULL,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_calls (
        id              TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        tool_name       TEXT NOT NULL,
        input           TEXT,
        output          TEXT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS retrieval_results (
        id              TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        query           TEXT NOT NULL,
        sources         JSONB,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_conv_session   ON conversations(session_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_tool_conv      ON tool_calls(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_retrieval_conv ON retrieval_results(conversation_id)",
]


async def init_db(database_url: str) -> asyncpg.Pool:
    """Create connection pool and ensure tables exist. Returns the pool."""
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
    async with pool.acquire() as conn:
        for stmt in _DDL:
            await conn.execute(stmt)
    return pool


class ConversationRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def ensure_session(self, session_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sessions (id) VALUES ($1) ON CONFLICT DO NOTHING",
                session_id,
            )
            await conn.execute(
                "UPDATE sessions SET updated_at = NOW() WHERE id = $1",
                session_id,
            )

    async def save_turn(self, session_id: str, role: str, content: str) -> str:
        """Insert a conversation turn and return its ID."""
        await self.ensure_session(session_id)
        turn_id = str(uuid.uuid4())
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversations (id, session_id, role, content)
                VALUES ($1, $2, $3, $4)
                """,
                turn_id,
                session_id,
                role,
                content,
            )
        return turn_id

    async def save_tool_call(
        self,
        conversation_id: str,
        tool_name: str,
        input: str,
        output: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tool_calls (id, conversation_id, tool_name, input, output)
                VALUES ($1, $2, $3, $4, $5)
                """,
                str(uuid.uuid4()),
                conversation_id,
                tool_name,
                input,
                output,
            )

    async def save_retrieval(
        self,
        conversation_id: str,
        query: str,
        sources: list[dict],
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO retrieval_results (id, conversation_id, query, sources)
                VALUES ($1, $2, $3, $4)
                """,
                str(uuid.uuid4()),
                conversation_id,
                query,
                json.dumps(sources),
            )

    async def get_history(self, session_id: str, max_turns: int = 10) -> list[dict]:
        """Return the last `max_turns` user+assistant pairs for a session."""
        limit = max_turns * 2
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content
                FROM conversations
                WHERE session_id = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                session_id,
                limit,
            )
        return [dict(r) for r in rows]

    async def close(self) -> None:
        await self._pool.close()
