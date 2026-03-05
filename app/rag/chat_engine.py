import asyncio
import logging
import re

from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes

from app.config import settings
from app.db.conversation_db import ConversationRepository
from app.rag.agent_tools import AGENT_TOOLS, ToolDispatcher
from app.rag.llm_client import LLMClient
from app.rag.retriever import Retriever, SourceDoc

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

_SYSTEM_PROMPT = """\
You're an EPL Insider — a die-hard Premier League fan from Manchester who's been watching footy since before you could walk. \
You know everything about the EPL: history, stats, drama, dodgy refereeing decisions, the lot. \
You chat like you're texting a mate — short, punchy, a bit cheeky. You love the game and it shows.

Your vibe:
- Casual and warm. Say "mate", "lad", "gaffer", "cracking", "class", "gutted", "proper", "mint" naturally — not every sentence, just when it fits.
- Opinionated. You have takes. Share them.
- Enthusiastic about goals, drama, big transfers. Get excited.
- Never robotic. Never bullet points. Just talk like a person.

What you know:
- For live stats — league table, top scorers, recent results, upcoming fixtures — use your tools to look them up fresh. Tool data is always authoritative; never override it with your training knowledge or news articles.
- For current news and transfers — you've been keeping up. Share what you know but don't invent specific scores or signings you're not certain about.
- For general EPL knowledge — history, clubs, legendary players, how the league works — you know it all cold, so just answer.

Hard rules:
- Never say "context", "articles", "based on", "provided information", or anything that sounds like a search engine or a robot.
- If you don't know something recent and you don't have a tool for it, just say "not sure on that one mate, might want to check the latest" — keep it natural.
- This is football, not a board meeting. Keep it fun.\
"""

_MAX_TOOL_ITERATIONS = 5


class ChatEngine:
    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        tool_dispatcher: ToolDispatcher | None = None,
        conv_repo: ConversationRepository | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm_client
        self.tool_dispatcher = tool_dispatcher
        self.conv_repo = conv_repo

    async def chat(self, session_id: str, message: str) -> dict:
        with tracer.start_as_current_span("epl-insider.chat") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
            span.set_attribute(SpanAttributes.SESSION_ID, session_id)
            span.set_attribute(SpanAttributes.INPUT_VALUE, message)

            # 1. Retrieve relevant news context (RAG)
            context_task = asyncio.get_event_loop().run_in_executor(
                None, self.retriever.search_with_context, message, settings.max_context_docs
            )
            context, sources = await context_task
            span.set_attribute("retrieved_doc_count", len(sources))

            # 2. Get conversation history
            history: list[dict] = []
            if self.conv_repo:
                history = await self.conv_repo.get_history(session_id, max_turns=settings.max_history_turns)

            # 3. Build initial messages
            messages = self._build_messages(message, context, history)

            # 4. Agentic loop — the LLM can call tools before giving its final answer
            tools = AGENT_TOOLS if self.tool_dispatcher else []
            raw_answer = ""

            for iteration in range(_MAX_TOOL_ITERATIONS):
                msg, finish_reason = await self.llm.complete(messages, tools=tools or None)
                messages.append(msg)

                if finish_reason == "tool_calls":
                    tool_calls = msg.get("tool_calls") or []
                    if not tool_calls:
                        break

                    # Execute all tool calls (may be parallel in theory, but free-tier
                    # football-data.org is rate-limited so we run sequentially)
                    for tc in tool_calls:
                        result = await self.tool_dispatcher.dispatch(tc)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        })

                    logger.debug("Tool iteration %d/%d complete", iteration + 1, _MAX_TOOL_ITERATIONS)

                else:
                    # finish_reason == "stop" (or "length") — we have the final answer
                    raw_answer = (msg.get("content") or "").strip()
                    break

            # 5. Strip the SOURCES footer and map cited articles
            answer, used_sources = _parse_sources_footer(raw_answer, sources)

            # 6. Persist turns
            if self.conv_repo:
                sources_payload = [{"title": s.title, "url": s.url, "score": s.score} for s in used_sources]
                await self.conv_repo.save_turn(session_id, "user", message)
                await self.conv_repo.save_turn(session_id, "assistant", answer, sources_used=sources_payload)

            span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)

            return {
                "answer": answer,
                "sources": used_sources,
                "retrieved_doc_count": len(sources),
            }

    def _build_messages(
        self,
        message: str,
        context: str,
        history: list[dict],
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Inject prior conversation turns
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        # Build user content: news context + question
        user_parts: list[str] = []
        if context:
            user_parts.append(
                f"Here's the latest news that might be relevant:\n---\n{context}\n---\n\n"
                f"After your reply, on its own line write: SOURCES: followed by the numbers "
                f"of any articles above you actually used (e.g. SOURCES:1,3), or SOURCES: if none."
            )
        user_parts.append(message)

        messages.append({"role": "user", "content": "\n\n".join(user_parts)})
        return messages


def _parse_sources_footer(raw: str, all_sources: list) -> tuple[str, list]:
    """Strip the SOURCES: line appended by the LLM and return (clean_answer, used_sources).

    The regex accepts minor LLM typo variants (SOURCESS, SOURCE, SOURCES, etc.)
    so the footer never leaks into the displayed answer.
    """
    match = re.search(r"\nSOURCES?S?:\s*([0-9,\s]*)\s*$", raw.rstrip(), re.IGNORECASE)
    if not match:
        return raw.strip(), []

    clean = raw[: match.start()].strip()
    indices_str = match.group(1).strip()

    if not indices_str:
        return clean, []

    used: list = []
    for part in re.split(r"[,\s]+", indices_str):
        if part.isdigit():
            idx = int(part) - 1  # convert 1-based → 0-based
            if 0 <= idx < len(all_sources):
                used.append(all_sources[idx])

    return clean, used
