import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.football-data.org/v4"


@dataclass
class _Cache:
    content: str
    fetched_at: float = field(default_factory=time.time)


class FootballDataClient:
    """Fetches live EPL stats from football-data.org (free tier).

    Uses 4 concurrent requests per refresh (standings, recent results,
    top scorers, upcoming fixtures). Results are cached for `cache_ttl_seconds`
    (default 10 min) so we stay well within the 10 req/min free-tier limit.
    """

    def __init__(self, api_key: str, cache_ttl_seconds: int = 600) -> None:
        self._ttl = cache_ttl_seconds
        self._cache: _Cache | None = None
        self._caches: dict[str, _Cache] = {}  # per-method caches
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={"X-Auth-Token": api_key},
            timeout=10.0,
        )

    async def get_formatted_stats(self) -> str | None:
        """Return a formatted stats string, using cache when fresh.

        Returns None on failure so the caller degrades gracefully.
        """
        now = time.time()
        if self._cache and (now - self._cache.fetched_at) < self._ttl:
            return self._cache.content

        try:
            content = await self._fetch_and_format()
            self._cache = _Cache(content=content)
            return content
        except Exception as exc:
            logger.warning("Live stats fetch failed: %s", exc)
            return self._cache.content if self._cache else None

    async def _fetch_and_format(self) -> str:
        today = datetime.now(timezone.utc).date()
        two_weeks_ago = (today - timedelta(days=14)).isoformat()
        three_weeks_ahead = (today + timedelta(days=21)).isoformat()

        (
            standings_resp,
            results_resp,
            scorers_resp,
            fixtures_resp,
        ) = await asyncio.gather(
            self._client.get("/competitions/PL/standings"),
            self._client.get(
                "/competitions/PL/matches",
                params={"status": "FINISHED", "dateFrom": two_weeks_ago, "dateTo": today.isoformat()},
            ),
            self._client.get("/competitions/PL/scorers", params={"limit": 15}),
            self._client.get(
                "/competitions/PL/matches",
                params={"status": "SCHEDULED", "dateFrom": today.isoformat(), "dateTo": three_weeks_ahead},
            ),
            return_exceptions=True,
        )

        parts: list[str] = []

        if isinstance(standings_resp, Exception):
            logger.warning("Standings request error: %s", standings_resp)
        else:
            standings_resp.raise_for_status()
            formatted = _format_standings(standings_resp.json())
            if formatted:
                parts.append(formatted)

        if isinstance(scorers_resp, Exception):
            logger.warning("Scorers request error: %s", scorers_resp)
        else:
            scorers_resp.raise_for_status()
            formatted = _format_top_scorers(scorers_resp.json())
            if formatted:
                parts.append(formatted)

        if isinstance(results_resp, Exception):
            logger.warning("Results request error: %s", results_resp)
        else:
            results_resp.raise_for_status()
            formatted = _format_recent_results(results_resp.json())
            if formatted:
                parts.append(formatted)

        if isinstance(fixtures_resp, Exception):
            logger.warning("Fixtures request error: %s", fixtures_resp)
        else:
            fixtures_resp.raise_for_status()
            formatted = _format_upcoming_fixtures(fixtures_resp.json())
            if formatted:
                parts.append(formatted)

        if not parts:
            raise RuntimeError("No stats data returned from football-data.org")

        return "\n\n".join(parts)

    # ── Individual callable methods (used by the agentic tool dispatcher) ──────

    async def get_standings(self) -> str:
        async def fetch() -> str:
            resp = await self._client.get("/competitions/PL/standings")
            resp.raise_for_status()
            return _format_standings(resp.json()) or "No standings data available."
        return await self._cached("standings", fetch)

    async def get_top_scorers(self, limit: int = 10) -> str:
        limit = min(max(limit, 1), 20)
        async def fetch() -> str:
            resp = await self._client.get("/competitions/PL/scorers", params={"limit": limit})
            resp.raise_for_status()
            return _format_top_scorers(resp.json()) or "No scorers data available."
        return await self._cached(f"scorers_{limit}", fetch)

    async def get_recent_results(self, days: int = 14) -> str:
        days = min(max(days, 1), 30)
        async def fetch() -> str:
            today = datetime.now(timezone.utc).date()
            date_from = (today - timedelta(days=days)).isoformat()
            resp = await self._client.get(
                "/competitions/PL/matches",
                params={"status": "FINISHED", "dateFrom": date_from, "dateTo": today.isoformat()},
            )
            resp.raise_for_status()
            return _format_recent_results(resp.json(), days=days) or f"No results found in the last {days} days."
        return await self._cached(f"results_{days}", fetch)

    async def get_upcoming_fixtures(self, days: int = 21) -> str:
        days = min(max(days, 1), 60)
        async def fetch() -> str:
            today = datetime.now(timezone.utc).date()
            date_to = (today + timedelta(days=days)).isoformat()
            resp = await self._client.get(
                "/competitions/PL/matches",
                params={"status": "SCHEDULED", "dateFrom": today.isoformat(), "dateTo": date_to},
            )
            resp.raise_for_status()
            return _format_upcoming_fixtures(resp.json(), days=days) or f"No fixtures found in the next {days} days."
        return await self._cached(f"fixtures_{days}", fetch)

    async def _cached(self, key: str, fetcher: Callable[[], Awaitable[str]]) -> str:
        now = time.time()
        cached = self._caches.get(key)
        if cached and (now - cached.fetched_at) < self._ttl:
            return cached.content
        try:
            content = await fetcher()
            self._caches[key] = _Cache(content=content)
            return content
        except Exception as exc:
            logger.warning("Stats fetch failed for '%s': %s", key, exc)
            if cached:
                return cached.content
            raise

    async def close(self) -> None:
        await self._client.aclose()


# ── Formatters ────────────────────────────────────────────────────────────────

def _format_standings(data: dict) -> str:
    try:
        table = next(
            s for s in data["standings"] if s["type"] == "TOTAL"
        )["table"]
    except (KeyError, StopIteration):
        return ""

    matchday = data.get("season", {}).get("currentMatchday", "?")
    lines = [f"PREMIER LEAGUE TABLE (Matchday {matchday})"]
    lines.append(f"{'Pos':<4} {'Team':<22} {'P':>2}  {'W':>2} {'D':>2} {'L':>2}  {'GF':>3} {'GA':>3} {'GD':>4}  {'Pts':>3}")
    lines.append("-" * 62)
    for row in table:
        pos = row["position"]
        team = row["team"].get("shortName") or row["team"]["name"]
        pts = row["points"]
        gd = row["goalDifference"]
        gd_str = f"+{gd}" if gd > 0 else str(gd)
        w, d, l_ = row["won"], row["draw"], row["lost"]
        played = row["playedGames"]
        gf = row.get("goalsFor", "-")
        ga = row.get("goalsAgainst", "-")
        lines.append(
            f"{pos:<4} {team:<22} {played:>2}  {w:>2} {d:>2} {l_:>2}  {gf:>3} {ga:>3} {gd_str:>4}  {pts:>3}"
        )

    return "\n".join(lines)


def _format_top_scorers(data: dict) -> str:
    scorers = data.get("scorers", [])
    if not scorers:
        return ""

    matchday = data.get("season", {}).get("currentMatchday", "?")
    lines = [f"TOP SCORERS (Matchday {matchday})"]
    for i, s in enumerate(scorers, 1):
        player = s["player"]["name"]
        team = s["team"].get("shortName") or s["team"]["name"]
        goals = s.get("goals") or 0
        assists = s.get("assists") or 0
        penalties = s.get("penalties") or 0
        pen_note = f" ({penalties} pens)" if penalties else ""
        lines.append(
            f"{i:>2}. {player} ({team}) — {goals}G{pen_note}, {assists}A"
        )

    return "\n".join(lines)


def _format_recent_results(data: dict, days: int = 14) -> str:
    matches = data.get("matches", [])
    if not matches:
        return ""

    sorted_matches = sorted(
        matches,
        key=lambda m: m.get("utcDate", ""),
        reverse=True,
    )

    lines = [f"RECENT RESULTS (last {days} days)"]
    for m in sorted_matches:
        home = m["homeTeam"].get("shortName") or m["homeTeam"]["name"]
        away = m["awayTeam"].get("shortName") or m["awayTeam"]["name"]
        score = m.get("score", {}).get("fullTime", {})
        hs = score.get("home", "?")
        as_ = score.get("away", "?")
        date = m.get("utcDate", "")[:10]
        matchday = m.get("matchday", "")
        md_note = f"  MD{matchday}" if matchday else ""
        lines.append(f"{home} {hs}–{as_} {away}  ({date}){md_note}")

    return "\n".join(lines)


def _format_upcoming_fixtures(data: dict, days: int = 21) -> str:
    matches = data.get("matches", [])
    if not matches:
        return ""

    sorted_matches = sorted(matches, key=lambda m: m.get("utcDate", ""))

    lines = [f"UPCOMING FIXTURES (next {days} days)"]
    for m in sorted_matches:
        home = m["homeTeam"].get("shortName") or m["homeTeam"]["name"]
        away = m["awayTeam"].get("shortName") or m["awayTeam"]["name"]
        utc = m.get("utcDate", "")
        # Format: "2025-02-26T20:00:00Z" → "Wed 26 Feb, 20:00"
        try:
            dt = datetime.fromisoformat(utc.replace("Z", "+00:00"))
            day_str = dt.strftime("%a %d %b, %H:%M UTC")
        except Exception:
            day_str = utc[:16].replace("T", " ") + " UTC"
        matchday = m.get("matchday", "")
        md_note = f"  (MD{matchday})" if matchday else ""
        lines.append(f"{home} vs {away}  —  {day_str}{md_note}")

    return "\n".join(lines)
