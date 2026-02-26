import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import settings
from app.rag.llm_client import LLMClient
from app.rag.retriever import Retriever, SourceDoc

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You're Terry — a die-hard Premier League fan from Manchester who's been watching footy since before you could walk. \
You know everything about the EPL: history, stats, drama, dodgy refereeing decisions, the lot. \
You chat like you're texting a mate — short, punchy, a bit cheeky. You love the game and it shows.

Your vibe:
- Casual and warm. Say "mate", "lad", "gaffer", "cracking", "class", "gutted", "proper", "mint" naturally — not every sentence, just when it fits.
- Opinionated. You have takes. Share them.
- Enthusiastic about goals, drama, big transfers. Get excited.
- Never robotic. Never bullet points. Just talk like a person.

What you know:
- For current news, results, transfers — you've been keeping up. Share what you know but don't invent specific scores or signings you're not certain about.
- For general EPL knowledge — history, clubs, legendary players, how the league works — you know it all cold, so just answer.

Hard rules:
- Never say "context", "articles", "based on", "provided information", or anything that sounds like a search engine or a robot.
- If you don't know something recent, just say "not sure on that one mate, might want to check the latest" — keep it natural.
- This is football, not a board meeting. Keep it fun.\
"""


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationStore:
    def __init__(self, max_turns: int = 5) -> None:
        self._sessions: dict[str, list[Turn]] = defaultdict(list)
        self.max_turns = max_turns

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        turns = self._sessions[session_id]
        turns.append(Turn(role=role, content=content))
        # Keep only the last max_turns pairs (user + assistant = 2 entries each)
        cap = self.max_turns * 2
        if len(turns) > cap:
            self._sessions[session_id] = turns[-cap:]

    def get_history(self, session_id: str) -> list[Turn]:
        return self._sessions.get(session_id, [])

    def format_history(self, session_id: str) -> str:
        turns = self.get_history(session_id)
        if not turns:
            return ""
        lines = []
        for turn in turns:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.content}")
        return "\n".join(lines)


class ChatEngine:
    def __init__(self, retriever: Retriever, llm_client: LLMClient) -> None:
        self.retriever = retriever
        self.llm = llm_client
        self.history = ConversationStore(max_turns=settings.max_history_turns)

    async def chat(self, session_id: str, message: str) -> dict:
        # 1. Retrieve relevant context
        context, sources = self.retriever.search_with_context(
            message, top_k=settings.max_context_docs
        )

        # 2. Get conversation history
        history = self.history.get_history(session_id)

        # 3. Build messages list for the Chat Completions API
        messages = self._build_messages(message, context, history)

        # 4. Generate answer
        answer = await self.llm.generate(messages)

        # 5. Store turns
        self.history.add_turn(session_id, "user", message)
        self.history.add_turn(session_id, "assistant", answer)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_doc_count": len(sources),
        }

    def _build_messages(self, message: str, context: str, history: list[Turn]) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Inject prior conversation turns so the model has context
        for turn in history:
            messages.append({"role": turn.role, "content": turn.content})

        # Current user message with retrieved context prepended
        user_content = (
            f"Here's the latest news that might be relevant:\n"
            f"---\n{context}\n---\n\n"
            f"{message}"
        )
        messages.append({"role": "user", "content": user_content})

        return messages
