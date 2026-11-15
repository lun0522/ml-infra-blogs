"""YAML configuration loader and validator for sources.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import SourceConfig

VALID_MODES = {"rss", "sitemap", "html_links"}
MODE_REQUIRED_FIELDS = {
    "rss": "feed_url",
    "sitemap": "sitemap_url",
    "html_links": "index_url",
}


def load_config(path: Path) -> list[SourceConfig]:
    """Load and validate blog sources from a YAML file."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    sources: list[SourceConfig] = []
    seen_ids: set[str] = set()
    for entry in raw.get("sources", []):
        source = _parse_source(entry, seen_ids)
        seen_ids.add(source.id)
        sources.append(source)

    return sources


def _parse_source(
    entry: dict[str, Any],
    seen_ids: set[str],
) -> SourceConfig:
    source_id = str(entry["id"]).strip()
    if not source_id:
        raise ValueError("Source entry is missing a non-empty 'id'")
    if source_id in seen_ids:
        raise ValueError(f"Duplicate source id '{source_id}' in config")

    mode = str(entry.get("mode", "")).strip()
    if mode not in VALID_MODES:
        raise ValueError(f"Source '{source_id}' has unsupported mode '{mode}'")

    required_field = MODE_REQUIRED_FIELDS[mode]
    if not entry.get(required_field):
        raise ValueError(
            f"Source '{source_id}' with mode '{mode}' must define '{required_field}'"
        )

    return SourceConfig(
        id=source_id,
        name=str(entry["name"]).strip(),
        homepage=str(entry["homepage"]).strip(),
        mode=mode,
        feed_url=_opt_str(entry.get("feed_url")),
        sitemap_url=_opt_str(entry.get("sitemap_url")),
        index_url=_opt_str(entry.get("index_url")),
        url_pattern=_opt_str(entry.get("url_pattern")),
        exclude_pattern=_opt_str(entry.get("exclude_pattern")),
        sitemap_pattern=_opt_str(entry.get("sitemap_pattern")),
        max_items=int(entry.get("max_items", 50)),
    )


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
