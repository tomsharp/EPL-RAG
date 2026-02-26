"""OpenAI function-tool definitions and dispatcher for Footy Phil's agentic mode.

The LLM decides which tools to call based on the user's question; this module
handles the schema declarations and routes each tool_call to the correct
FootballDataClient method.
"""

import json
import logging

logger = logging.getLogger(__name__)

# ── Tool schemas passed to OpenAI chat completions ────────────────────────────

AGENT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_standings",
            "description": (
                "Fetch the current Premier League table showing position, points, "
                "wins, draws, losses, goals for/against, and goal difference for all 20 clubs."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_scorers",
            "description": (
                "Fetch the top goal scorers in the current Premier League season, "
                "including goals, assists, and penalty goals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "How many top scorers to return (default 10, max 20).",
                        "default": 10,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_results",
            "description": (
                "Fetch Premier League match results (finished games) from the past N days. "
                "Use this when the user asks about recent scores, results, or how a team has been doing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "How many days back to search for results (default 14, max 30).",
                        "default": 14,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_upcoming_fixtures",
            "description": (
                "Fetch upcoming Premier League fixtures (scheduled matches) in the next N days. "
                "Use this when the user asks about upcoming games, next match, fixture list, or schedules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "How many days ahead to look for fixtures (default 21, max 60).",
                        "default": 21,
                    }
                },
                "required": [],
            },
        },
    },
]


# ── Dispatcher ────────────────────────────────────────────────────────────────

class ToolDispatcher:
    """Routes OpenAI tool_call objects to FootballDataClient methods."""

    def __init__(self, stats_client) -> None:
        self._client = stats_client

    async def dispatch(self, tool_call: dict) -> str:
        """Execute a single tool call and return the result as a string."""
        name = tool_call["function"]["name"]
        try:
            args: dict = json.loads(tool_call["function"].get("arguments") or "{}")
        except (json.JSONDecodeError, KeyError):
            args = {}

        logger.info("Tool call: %s(%s)", name, args)

        try:
            if name == "get_standings":
                return await self._client.get_standings()

            elif name == "get_top_scorers":
                limit = int(args.get("limit", 10))
                return await self._client.get_top_scorers(limit=limit)

            elif name == "get_recent_results":
                days = int(args.get("days", 14))
                return await self._client.get_recent_results(days=days)

            elif name == "get_upcoming_fixtures":
                days = int(args.get("days", 21))
                return await self._client.get_upcoming_fixtures(days=days)

            else:
                logger.warning("Unknown tool called: %s", name)
                return f"Unknown tool: {name}"

        except Exception as exc:
            logger.error("Tool %s failed: %s", name, exc)
            return f"Could not fetch data for {name}: {exc}"
