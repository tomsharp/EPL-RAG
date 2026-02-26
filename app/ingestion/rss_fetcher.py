import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import feedparser
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

RSS_FEEDS: dict[str, str] = {
    "bbc": "https://feeds.bbci.co.uk/sport/football/premier-league/rss.xml",
    "guardian": "https://www.theguardian.com/football/premierleague/rss",
    "skysports": "https://www.skysports.com/rss/11095",  # Premier League-specific
    "football365": "https://www.football365.com/feed",
    "90min": "https://www.90min.com/posts.rss",
    "goal": "https://www.goal.com/feeds/en/news",
    "talksport": "https://talksport.com/football/feed/",
    "mirror": "https://www.mirror.co.uk/sport/football/rss.xml",
    "express": "https://www.express.co.uk/sport/football/rss",
}


@dataclass
class Article:
    url: str
    title: str
    summary: str
    published: Optional[datetime]
    source: str
    content_hash: str


class RSSFetcher:
    def __init__(self, feeds: dict[str, str] = RSS_FEEDS) -> None:
        self.feeds = feeds

    def fetch_all(self) -> list[Article]:
        articles: list[Article] = []
        seen_urls: set[str] = set()

        for source, url in self.feeds.items():
            try:
                fetched = self._fetch_feed(source, url)
                for article in fetched:
                    if article.url not in seen_urls:
                        seen_urls.add(article.url)
                        articles.append(article)
            except Exception as exc:
                logger.warning("Failed to fetch feed '%s' (%s): %s", source, url, exc)

        logger.info("Fetched %d unique articles across %d feeds", len(articles), len(self.feeds))
        return articles

    def _fetch_feed(self, source: str, url: str) -> list[Article]:
        feed = feedparser.parse(url)
        articles: list[Article] = []

        for entry in feed.entries:
            article = self._parse_entry(entry, source)
            if article is not None:
                articles.append(article)

        logger.debug("Feed '%s': parsed %d articles", source, len(articles))
        return articles

    def _parse_entry(self, entry: feedparser.FeedParserDict, source: str) -> Optional[Article]:
        try:
            url = entry.get("link", "").strip()
            title = self._clean_text(entry.get("title", ""))

            # Guardian provides full body in content:encoded; fall back to summary
            content_list = entry.get("content", [])
            if content_list:
                raw_summary = content_list[0].get("value", "")
            else:
                raw_summary = entry.get("summary", "")

            summary = self._clean_text(raw_summary)

            if not url or not title or not summary:
                return None

            # Truncate summary to 1000 chars to keep payloads manageable
            summary = summary[:1000]

            published = self._parse_date(entry)
            content_hash = self._make_hash(url, title, summary)

            return Article(
                url=url,
                title=title,
                summary=summary,
                published=published,
                source=source,
                content_hash=content_hash,
            )
        except Exception as exc:
            logger.debug("Skipping malformed entry: %s", exc)
            return None

    def _parse_date(self, entry: feedparser.FeedParserDict) -> Optional[datetime]:
        # feedparser populates published_parsed as a time.struct_time
        if entry.get("published_parsed"):
            try:
                import calendar
                ts = calendar.timegm(entry.published_parsed)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                pass
        if entry.get("published"):
            try:
                return parsedate_to_datetime(entry.published)
            except Exception:
                pass
        return None

    def _make_hash(self, url: str, title: str, summary: str) -> str:
        content = f"{url}|{title}|{summary[:500]}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _clean_text(self, raw: str) -> str:
        if not raw:
            return ""
        # Strip HTML tags
        soup = BeautifulSoup(raw, "lxml")
        text = soup.get_text(separator=" ")
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
