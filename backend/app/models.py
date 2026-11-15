"""Data models for configured blog sources, posts, and archive exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SourceConfig:
    """Declarative monitor configuration for a single blog or feed."""

    id: str
    name: str
    homepage: str
    mode: str  # "rss" | "sitemap" | "html_links"
    feed_url: str | None = None
    sitemap_url: str | None = None
    index_url: str | None = None
    url_pattern: str | None = None
    exclude_pattern: str | None = None
    sitemap_pattern: str | None = None
    max_items: int = 50

    @property
    def monitor_url(self) -> str:
        """Primary endpoint fetched during daily monitoring."""
        return self.feed_url or self.sitemap_url or self.index_url or self.homepage


@dataclass
class RawPost:
    """Intermediate extracted post before merging into the persistent archive."""

    url: str
    title: str
    published_at: str | None = None
    summary: str = ""
    needs_enrichment: bool = False


@dataclass
class PostRecord:
    """Canonical persisted post entry stored in data/posts.json."""

    id: str
    source_id: str
    source_name: str
    url: str
    title: str
    summary: str
    published_at: str | None
    first_seen: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceHealth:
    """Per-source fetch outcome recorded on each update run."""

    id: str
    name: str
    homepage: str
    mode: str
    monitor_url: str
    status: str  # "ok" | "error"
    error: str | None
    fetched_count: int
    new_count: int
    total_count: int
    last_checked: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
