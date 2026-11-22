"""Archive persistence, concurrent source refresh, and static JSON export."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .classifier import load_default_classifier
from .config import load_config
from .extractors import (
    MIN_ENRICHED_EXCERPT_LEN,
    build_http_client,
    canonical_key,
    extract_article_details,
    fetch_source_posts,
    is_lead_skip_text,
    post_id_for_url,
    should_upgrade_summary,
)
from .models import PostRecord, RawPost, SourceConfig, SourceHealth

DEFAULT_MAX_ARCHIVE_POSTS = 2000


def utc_now_iso() -> str:
    """Current UTC timestamp formatted as ISO-8601."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_archive(path: Path) -> dict[str, Any]:
    """Load an existing posts.json archive or return an empty baseline."""
    if not path.exists():
        return {"generated_at": None, "last_new_count": 0, "sources": [], "posts": []}
    return json.loads(path.read_text(encoding="utf-8"))


def run_update(
    config_path: Path,
    archive_path: Path,
    export_dir: Path | None = None,
    from_archive: bool = False,
    refresh_thin_excerpts: bool = False,
    reclassify: bool = False,
    max_workers: int = 8,
    http_get: Callable[[str], str] | None = None,
    classifier: Any = None,
) -> dict[str, Any]:
    """Refresh all sources (unless from_archive=True), merge archive, and write static exports."""
    sources = load_config(config_path)
    existing_archive = load_archive(archive_path)
    existing_posts = _index_existing_posts(existing_archive.get("posts", []))

    if from_archive:
        if reclassify and existing_posts:
            _classify_posts(list(existing_posts.values()), classifier)
        payload = _rebuild_from_existing(sources, existing_archive, existing_posts)
    else:
        payload = _fetch_and_merge(
            sources=sources,
            existing_posts=existing_posts,
            refresh_thin_excerpts=refresh_thin_excerpts,
            reclassify=reclassify,
            max_workers=max_workers,
            http_get=http_get,
            classifier=classifier,
        )

    _write_json(archive_path, payload)
    if export_dir is not None:
        _write_json(export_dir / "posts.json", payload)

    return payload


def _index_existing_posts(raw_posts: list[dict[str, Any]]) -> dict[str, PostRecord]:
    indexed: dict[str, PostRecord] = {}
    for item in raw_posts:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        key = canonical_key(url)
        indexed[key] = PostRecord(
            id=str(item.get("id") or post_id_for_url(url)),
            source_id=str(item.get("source_id", "")),
            source_name=str(item.get("source_name", "")),
            url=url,
            title=str(item.get("title") or url),
            summary=str(item.get("summary") or ""),
            published_at=item.get("published_at"),
            first_seen=str(item.get("first_seen") or utc_now_iso()),
            label=str(item.get("label") or "infra"),
            topics=[str(t) for t in (item.get("topics") or [])],
            excerpt=str(item.get("excerpt") or ""),
        )
    return indexed


def _rebuild_from_existing(
    sources: list[SourceConfig],
    existing_archive: dict[str, Any],
    existing_posts: dict[str, PostRecord],
) -> dict[str, Any]:
    prior_sources = {s["id"]: s for s in existing_archive.get("sources", []) if "id" in s}
    posts_list = _trim_posts(list(existing_posts.values()), sources)
    counts_by_source = _count_posts_by_source(posts_list)

    source_health: list[dict[str, Any]] = []
    for src in sources:
        prev = prior_sources.get(src.id, {})
        source_health.append(
            SourceHealth(
                id=src.id,
                name=src.name,
                homepage=src.homepage,
                mode=src.mode,
                monitor_url=src.monitor_url,
                status=str(prev.get("status", "ok")),
                error=prev.get("error"),
                fetched_count=int(prev.get("fetched_count", 0)),
                new_count=int(prev.get("new_count", 0)),
                total_count=counts_by_source.get(src.id, 0),
                last_checked=str(prev.get("last_checked") or existing_archive.get("generated_at") or utc_now_iso()),
            ).to_dict()
        )

    return {
        "generated_at": existing_archive.get("generated_at") or utc_now_iso(),
        "last_new_count": int(existing_archive.get("last_new_count", 0)),
        "sources": source_health,
        "posts": [p.to_dict() for p in posts_list],
    }


def _fetch_and_merge(
    sources: list[SourceConfig],
    existing_posts: dict[str, PostRecord],
    refresh_thin_excerpts: bool,
    reclassify: bool,
    max_workers: int,
    http_get: Callable[[str], str] | None,
    classifier: Any = None,
) -> dict[str, Any]:
    run_ts = utc_now_iso()
    seen_keys = {
        key
        for key, post in existing_posts.items()
        if post.published_at
        and post.summary
        and post.excerpt
        and (
            not refresh_thin_excerpts
            or (
                len(post.excerpt) >= MIN_ENRICHED_EXCERPT_LEN
                and not is_lead_skip_text(post.summary)
            )
        )
    }
    source_results: dict[str, tuple[list[RawPost], str | None]]

    def _execute(getter: Callable[[str], str]) -> tuple[dict[str, tuple[list[RawPost], str | None]], set[str]]:
        results = _fetch_with_custom_getter(sources, seen_keys, getter, max_workers)
        merged = {
            canonical_key(raw.url)
            for src in sources
            for raw in results[src.id][0]
        }
        return results, merged

    merged_keys: set[str] = set()
    if http_get is not None:
        source_results, merged_keys = _execute(http_get)
        total_new, new_records, health_by_id = _apply_source_results(
            sources, source_results, existing_posts, run_ts
        )
        _backfill_archived_excerpts(
            existing_posts=existing_posts,
            skip_keys=merged_keys,
            get_text=http_get,
            max_workers=max_workers,
            refresh_thin_excerpts=refresh_thin_excerpts,
        )
    else:
        with build_http_client() as client:
            def client_get(url: str) -> str:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.text

            source_results, merged_keys = _execute(client_get)
            total_new, new_records, health_by_id = _apply_source_results(
                sources, source_results, existing_posts, run_ts
            )
            _backfill_archived_excerpts(
                existing_posts=existing_posts,
                skip_keys=merged_keys,
                get_text=client_get,
                max_workers=max_workers,
                refresh_thin_excerpts=refresh_thin_excerpts,
            )

    # Refresh source names on existing posts in case a source was renamed.
    source_map = {src.id: src for src in sources}
    for post in existing_posts.values():
        if post.source_id in source_map:
            post.source_name = source_map[post.source_id].name

    to_classify = list(existing_posts.values()) if reclassify else new_records
    if to_classify:
        _classify_posts(to_classify, classifier)

    sorted_posts = _trim_posts(list(existing_posts.values()), sources)
    counts = _count_posts_by_source(sorted_posts)
    health_list = []
    for src in sources:
        h = health_by_id[src.id]
        h.total_count = counts.get(src.id, 0)
        health_list.append(h.to_dict())

    return {
        "generated_at": run_ts,
        "last_new_count": total_new,
        "sources": health_list,
        "posts": [p.to_dict() for p in sorted_posts],
    }


def _classify_posts(posts: list[PostRecord], classifier: Any = None) -> None:
    """Assign 5-pillar topic tags (empty topics => 'drop') using the active ONNX classifier."""
    active_classifier = classifier if classifier is not None else load_default_classifier()
    if active_classifier is None or not posts:
        return
    predictions = active_classifier.predict_batch([(p.title, p.excerpt or p.summary) for p in posts])
    for post, pred in zip(posts, predictions, strict=True):
        post.topics = list(pred.topics)
        post.label = pred.label


def _fetch_with_custom_getter(
    sources: list[SourceConfig],
    seen_keys: set[str],
    get_text: Callable[[str], str],
    max_workers: int,
) -> dict[str, tuple[list[RawPost], str | None]]:
    results: dict[str, tuple[list[RawPost], str | None]] = {}
    worker_count = max(1, min(max_workers, len(sources) or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_map = {
            pool.submit(fetch_source_posts, get_text, src, seen_keys): src
            for src in sources
        }
        for future in as_completed(future_map):
            src = future_map[future]
            try:
                posts = future.result()
                results[src.id] = (posts, None)
            except Exception as exc:
                results[src.id] = ([], f"{type(exc).__name__}: {exc}")
    return results


def _apply_source_results(
    sources: list[SourceConfig],
    source_results: dict[str, tuple[list[RawPost], str | None]],
    existing_posts: dict[str, PostRecord],
    run_ts: str,
) -> tuple[int, list[PostRecord], dict[str, SourceHealth]]:
    total_new = 0
    new_records: list[PostRecord] = []
    health_by_id: dict[str, SourceHealth] = {}
    for src in sources:
        raw_posts, err = source_results[src.id]
        new_for_source = 0
        for raw in raw_posts:
            key = canonical_key(raw.url)
            if key in existing_posts:
                _merge_existing(existing_posts[key], raw, src)
            else:
                record = PostRecord(
                    id=post_id_for_url(raw.url),
                    source_id=src.id,
                    source_name=src.name,
                    url=raw.url,
                    title=raw.title,
                    summary=raw.summary,
                    published_at=raw.published_at,
                    first_seen=run_ts,
                    excerpt=raw.excerpt or raw.summary,
                )
                existing_posts[key] = record
                new_records.append(record)
                new_for_source += 1
        total_new += new_for_source
        health_by_id[src.id] = SourceHealth(
            id=src.id,
            name=src.name,
            homepage=src.homepage,
            mode=src.mode,
            monitor_url=src.monitor_url,
            status="error" if err else "ok",
            error=err,
            fetched_count=len(raw_posts),
            new_count=new_for_source,
            total_count=0,
            last_checked=run_ts,
        )
    return total_new, new_records, health_by_id


def _backfill_archived_excerpts(
    existing_posts: dict[str, PostRecord],
    skip_keys: set[str],
    get_text: Callable[[str], str],
    max_workers: int,
    refresh_thin_excerpts: bool,
) -> None:
    """Enrich older archived posts that rolled off live feeds without an excerpt."""
    targets = [
        post
        for key, post in existing_posts.items()
        if key not in skip_keys
        and (
            not post.excerpt
            or (
                refresh_thin_excerpts
                and (
                    len(post.excerpt) < MIN_ENRICHED_EXCERPT_LEN
                    or is_lead_skip_text(post.summary)
                )
            )
        )
    ]
    if not targets:
        return

    def _enrich_one(post: PostRecord) -> tuple[PostRecord, str | None, str | None, str | None, str] | None:
        try:
            html_text = get_text(post.url)
        except Exception:
            return None
        meta_title, meta_date, meta_summary, meta_excerpt = extract_article_details(
            html_text, post.url
        )
        return post, meta_title, meta_date, meta_summary, meta_excerpt

    worker_count = max(1, min(max_workers, len(targets)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for res in pool.map(_enrich_one, targets):
            if res is None:
                continue
            post, meta_title, meta_date, meta_summary, meta_excerpt = res
            if meta_title and not post.title:
                post.title = meta_title
            if meta_date and not post.published_at:
                post.published_at = meta_date
            if should_upgrade_summary(post.summary, meta_summary, post.title):
                assert meta_summary is not None
                post.summary = meta_summary
            if meta_excerpt and len(meta_excerpt) > len(post.excerpt):
                post.excerpt = meta_excerpt
            elif not post.excerpt and post.summary:
                post.excerpt = post.summary


def _merge_existing(existing: PostRecord, raw: RawPost, src: SourceConfig) -> None:
    """Enrich missing fields on an already-archived post without altering first_seen."""
    existing.source_name = src.name
    if not raw.needs_enrichment and raw.title:
        existing.title = raw.title
    if raw.published_at and not existing.published_at:
        existing.published_at = raw.published_at
    if should_upgrade_summary(existing.summary, raw.summary, existing.title):
        existing.summary = raw.summary
    if raw.excerpt and len(raw.excerpt) > len(existing.excerpt):
        existing.excerpt = raw.excerpt


def _trim_posts(posts: list[PostRecord], sources: list[SourceConfig]) -> list[PostRecord]:
    """Sort newest-first, capping each source at max_items and total at DEFAULT_MAX_ARCHIVE_POSTS."""
    limits = {src.id: src.max_items for src in sources}
    kept: list[PostRecord] = []
    counts: dict[str, int] = {}
    for post in _sort_posts(posts):
        limit = limits.get(post.source_id)
        seen = counts.get(post.source_id, 0)
        if limit is not None and seen >= limit:
            continue
        counts[post.source_id] = seen + 1
        kept.append(post)
        if len(kept) >= DEFAULT_MAX_ARCHIVE_POSTS:
            break
    return kept


def _sort_posts(posts: list[PostRecord]) -> list[PostRecord]:
    return sorted(
        posts,
        key=lambda p: (p.published_at or p.first_seen, p.first_seen, p.url),
        reverse=True,
    )


def _count_posts_by_source(posts: list[PostRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for post in posts:
        counts[post.source_id] = counts.get(post.source_id, 0) + 1
    return counts


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
