"""Extractors for RSS/Atom feeds, XML sitemaps, and static HTML blog indexes."""

from __future__ import annotations

import email.utils
import hashlib
import html
import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Callable

import feedparser
import httpx
from bs4 import BeautifulSoup, Tag

from .models import RawPost, SourceConfig

USER_AGENT = (
    "Mozilla/5.0 (compatible; MLInfraBlogsBot/1.0; +https://github.com/lun0522/ml-infra-blogs)"
)
TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "source",
    "fbclid",
    "gclid",
}

DATE_TEXT_PATTERNS = [
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %b, %Y",
    "%d %B, %Y",
    "%d %b %Y",
    "%d %B %Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
]
INLINE_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*,?\s+\d{4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
URL_YMD_RE = re.compile(r"/(20\d{2})[-/](\d{2})[-/](\d{2})(?:[-/]|$)")
URL_YM_RE = re.compile(r"/(20\d{2})/(\d{2})(?:/|$)")
SUMMARY_MAX_LEN = 420
EXCERPT_MAX_LEN = 1600
STUB_SUMMARY_MAX_LEN = 160
MIN_ENRICHED_EXCERPT_LEN = 350
SKIP_PROSE_PARENTS = {"header", "nav", "footer", "aside", "figcaption", "form"}
LEAF_NOISE_RE = re.compile(
    r"(?:^|[-_\s])(?:newsletter|subscribe|cookie|consent|katex-mathml|related-posts|social-share|share-buttons|modal|popup)(?:$|[-_\s])",
    re.I,
)
BOILERPLATE_PHRASES = (
    "requires javascript",
    "enable javascript",
    "rotate your device",
    "sign up for our newsletter",
    "delivered directly to your inbox",
    "unsubscribe at anytime",
    "thanks for signing up",
    "your data is handled in accordance",
    "privacy policy",
    "cookie policy",
    "accept all cookies",
    "all rights reserved",
)
LEAD_SKIP_RE = re.compile(
    r"^(?:special thanks to\b|thanks to .+?\bfor (?:feedback|review|edits?|comments?)|authors\s*[—:-]|this form requires javascript)",
    re.I,
)
CHROME_HEADINGS = {
    "ai-generated summary",
    "next steps",
    "table of contents",
    "contents",
    "references",
    "footnotes",
    "acknowledgments",
    "acknowledgements",
    "resources",
    "featured",
    "related posts",
    "related tags",
    "read more",
    "share this post",
    "subscribe",
    "join the community",
    "join the community and follow along!",
}
CHROME_HEADING_SUBSTRS = (
    "read more from",
    "sign up for",
    "subscribe to receive",
    "newsletter",
)


def normalize_url(url: str) -> str:
    """Strip fragments and tracking query params while preserving canonical path."""
    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    query_pairs = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
        if k.lower() not in TRACKING_QUERY_KEYS
    ]
    query = urllib.parse.urlencode(query_pairs)
    path = parsed.path or "/"
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def canonical_key(url: str) -> str:
    """Stable deduplication key ignoring trailing-slash differences."""
    norm = normalize_url(url)
    parsed = urllib.parse.urlsplit(norm)
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def post_id_for_url(url: str) -> str:
    """Deterministic short ID for a canonical post URL."""
    return hashlib.sha1(canonical_key(url).encode("utf-8")).hexdigest()[:16]


def parse_date(value: str | None) -> str | None:
    """Parse common ISO, RFC-2822, and human date strings into UTC ISO-8601."""
    if not value:
        return None
    text = " ".join(html.unescape(value).strip().split())
    if not text:
        return None

    # Normalize full-month / Sept variants before strptime.
    text = re.sub(r"\bSept\b", "Sep", text, flags=re.IGNORECASE)

    # 1. ISO-8601 (handles 'Z' and timezone offsets)
    iso_candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(iso_candidate)
        return _to_utc_iso(dt)
    except ValueError:
        pass

    # 2. RFC-2822 (RSS pubDate)
    try:
        dt = email.utils.parsedate_to_datetime(text)
        if dt is not None:
            return _to_utc_iso(dt)
    except (TypeError, ValueError):
        pass

    # 3. Common human date formats (exact or extracted substring)
    candidates = [text]
    match = INLINE_DATE_RE.search(text)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        normalized = re.sub(r"\bSept\b", "Sep", candidate, flags=re.IGNORECASE)
        for fmt in DATE_TEXT_PATTERNS:
            try:
                dt = datetime.strptime(normalized, fmt)
                return _to_utc_iso(dt)
            except ValueError:
                continue

    return None


def struct_time_to_iso(st: time.struct_time | None) -> str | None:
    """Convert feedparser struct_time to UTC ISO-8601."""
    if not st:
        return None
    try:
        dt = datetime(*st[:6], tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None


def extract_date_from_url(url: str) -> str | None:
    """Infer publish date from URL slugs such as /blog/2026-11-13-title or /2025/01/16/."""
    match = URL_YMD_RE.search(url)
    if match:
        y, m, d = map(int, match.groups())
        try:
            return datetime(y, m, d, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None

    match_ym = URL_YM_RE.search(url)
    if match_ym:
        y, m = map(int, match_ym.groups())
        try:
            return datetime(y, m, 1, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None
    return None


def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_text(raw: str | None, max_len: int = SUMMARY_MAX_LEN) -> str:
    """Strip HTML markup, collapse whitespace, and truncate cleanly at word boundaries."""
    if not raw:
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    text = " ".join(html.unescape(text).split())
    if max_len > 0 and len(text) > max_len:
        cut = text[:max_len].rsplit(" ", 1)[0].rstrip(".,;:- ")
        return f"{cut or text[:max_len]}…"
    return text


def url_matches_source(url: str, source: SourceConfig) -> bool:
    """Check whether a URL matches the source's include/exclude regex filters."""
    if source.url_pattern and not re.search(source.url_pattern, url):
        return False
    if source.exclude_pattern and re.search(source.exclude_pattern, url):
        return False
    return True


def parse_rss_entries(xml_text: str, source: SourceConfig) -> list[RawPost]:
    """Parse RSS 2.0 or Atom feed entries filtered by source configuration."""
    parsed = feedparser.parse(xml_text)
    posts: list[RawPost] = []
    seen_keys: set[str] = set()

    for entry in parsed.entries:
        link = entry.get("link")
        if not link:
            continue
        url = normalize_url(link)
        key = canonical_key(url)
        if key in seen_keys or not url_matches_source(url, source):
            continue
        seen_keys.add(key)

        title = clean_text(entry.get("title"), max_len=200) or url
        published_at = (
            struct_time_to_iso(entry.get("published_parsed"))
            or struct_time_to_iso(entry.get("updated_parsed"))
            or parse_date(entry.get("published"))
            or parse_date(entry.get("updated"))
            or extract_date_from_url(url)
        )
        raw_summary = entry.get("summary") or entry.get("description") or ""
        raw_content = ""
        if entry.get("content"):
            candidates = [str(item.get("value") or "") for item in entry["content"]]
            raw_content = max(candidates, key=len, default="")

        summary_source = raw_summary or raw_content
        summary = clean_text(summary_source, max_len=SUMMARY_MAX_LEN)
        excerpt = ""
        if raw_content:
            content_lead, excerpt = _extract_lead_and_excerpt_from_html(raw_content, title)
            if content_lead and _is_stub_text(summary, title, min_len=STUB_SUMMARY_MAX_LEN):
                if len(content_lead) > len(summary):
                    summary = content_lead
        if len(excerpt) < len(raw_summary) and len(raw_summary) > SUMMARY_MAX_LEN:
            _rs_lead, rich_summary_excerpt = _extract_lead_and_excerpt_from_html(raw_summary, title)
            if len(rich_summary_excerpt) > len(excerpt):
                excerpt = rich_summary_excerpt
        if not excerpt:
            excerpt = summary

        needs_enrichment = (
            _needs_article_enrichment(
                title=title,
                published_at=published_at,
                summary=summary,
                excerpt=excerpt,
                has_heading=bool(title and title != url),
            )
            or (not raw_content and len(excerpt.strip()) < 1000)
        )

        posts.append(
            RawPost(
                url=url,
                title=title,
                published_at=published_at,
                summary=summary,
                excerpt=excerpt,
                needs_enrichment=needs_enrichment,
            )
        )
        if len(posts) >= source.max_items:
            break

    return posts


def parse_sitemap_xml(xml_text: str) -> tuple[list[tuple[str, str | None]], list[str]]:
    """Return (url_entries, child_sitemap_urls) from a sitemap XML document."""
    root = ET.fromstring(xml_text)
    tag = root.tag.split("}")[-1].lower()

    if tag == "sitemapindex":
        child_sitemaps: list[str] = []
        for node in root:
            if node.tag.split("}")[-1].lower() != "sitemap":
                continue
            loc = _find_child_text(node, "loc")
            if loc:
                child_sitemaps.append(loc.strip())
        return [], child_sitemaps

    entries: list[tuple[str, str | None]] = []
    for node in root:
        if node.tag.split("}")[-1].lower() != "url":
            continue
        loc = _find_child_text(node, "loc")
        lastmod = _find_child_text(node, "lastmod")
        if loc:
            entries.append((normalize_url(loc), parse_date(lastmod)))
    return entries, []


def _find_child_text(node: ET.Element, child_name: str) -> str | None:
    for child in node:
        if child.tag.split("}")[-1].lower() == child_name:
            return child.text
    return None


def parse_html_links(html_text: str, base_url: str, source: SourceConfig) -> list[RawPost]:
    """Extract candidate article links and card metadata from an HTML index page."""
    soup = BeautifulSoup(html_text, "html.parser")
    posts: list[RawPost] = []
    seen_keys: set[str] = set()
    index_key = canonical_key(base_url)

    for anchor in soup.find_all("a", href=True):
        raw_href = str(anchor["href"]).strip()
        if not raw_href or raw_href.startswith(("#", "mailto:", "javascript:")):
            continue
        url = normalize_url(urllib.parse.urljoin(base_url, raw_href))
        key = canonical_key(url)
        if key == index_key or key in seen_keys:
            continue
        if not url_matches_source(url, source):
            continue

        title, published_at, summary, has_heading = _extract_card_context(
            anchor, url, base_url, source
        )
        seen_keys.add(key)
        resolved_title = title or _slug_to_title(url)
        needs_enrichment = _needs_article_enrichment(
            title=resolved_title,
            published_at=published_at,
            summary=summary,
            excerpt=summary,
            has_heading=has_heading and bool(title),
        )
        posts.append(
            RawPost(
                url=url,
                title=resolved_title,
                published_at=published_at,
                summary=summary,
                excerpt=summary,
                needs_enrichment=needs_enrichment,
            )
        )
        if len(posts) >= source.max_items:
            break

    return posts


def _extract_card_context(
    anchor: Tag,
    url: str,
    base_url: str,
    source: SourceConfig,
) -> tuple[str, str | None, str, bool]:
    """Pull heading title, date, summary, and heading presence from an anchor or its card."""
    container = _find_card_container(anchor, base_url, source)

    # Prefer a heading inside the anchor, then inside its immediate card container.
    heading = anchor.find(["h1", "h2", "h3", "h4"]) or container.find(["h1", "h2", "h3", "h4"])
    title = clean_text(heading.get_text(" ", strip=True), max_len=200) if heading else ""

    # Date from <time>, card text, or URL slug.
    published_at: str | None = extract_date_from_url(url)
    time_tag = anchor.find("time") or container.find("time")
    if time_tag:
        published_at = (
            parse_date(str(time_tag.get("datetime") or ""))
            or parse_date(time_tag.get_text(" ", strip=True))
            or published_at
        )

    anchor_text = clean_text(anchor.get_text(" ", strip=True), max_len=260)
    if not published_at:
        card_text = clean_text(container.get_text(" ", strip=True), max_len=400)
        date_match = INLINE_DATE_RE.search(anchor_text) or INLINE_DATE_RE.search(card_text)
        if date_match:
            published_at = parse_date(date_match.group(0))

    if not title:
        # Remove inline date/read-time suffixes if anchor wraps an entire card without heading tags.
        cleaned_anchor = INLINE_DATE_RE.sub("", anchor_text)
        cleaned_anchor = re.sub(r"\b\d+\s*min\s*read\b", "", cleaned_anchor, flags=re.I)
        cleaned_anchor = re.sub(r"^(?:Featured|Announcement|Research|Perspective)\b[:\s]*", "", cleaned_anchor)
        cleaned_anchor = " ".join(cleaned_anchor.split()).strip(" ·-|—")
        if 4 <= len(cleaned_anchor) <= 160 and cleaned_anchor.lower() not in {"read more", "learn more"}:
            title = cleaned_anchor

    summary = ""
    for para in (*anchor.find_all("p"), *container.find_all("p")):
        para_text = clean_text(para.get_text(" ", strip=True), max_len=SUMMARY_MAX_LEN)
        if (
            len(para_text) >= 45
            and para_text != title
            and (not title or para_text.lower() not in title.lower())
            and not parse_date(para_text)
        ):
            summary = para_text
            break

    return title, published_at, summary, heading is not None


def _contains_multiple_cards(parent: Tag, base_url: str, source: SourceConfig) -> bool:
    """Return True once a wrapper spans more than one post card."""
    article_keys: set[str] = set()
    distinct_hrefs: set[str] = set()
    for link in parent.find_all("a", href=True):
        raw_href = str(link["href"]).strip()
        if not raw_href or raw_href.startswith(("#", "mailto:", "javascript:")):
            continue
        abs_url = normalize_url(urllib.parse.urljoin(base_url, raw_href))
        key = canonical_key(abs_url)
        distinct_hrefs.add(key)
        if url_matches_source(abs_url, source):
            article_keys.add(key)
        if len(article_keys) > 1 or len(distinct_hrefs) > 4:
            return True
    return False


def _find_card_container(anchor: Tag, base_url: str, source: SourceConfig) -> Tag:
    """Walk up at most 4 levels to find the enclosing wrapper for a single post card."""
    current: Tag = anchor
    for depth, parent in enumerate(anchor.parents):
        if not isinstance(parent, Tag) or depth >= 4:
            break
        if parent.name in {"body", "main", "section", "ul", "ol", "html"}:
            break
        if _contains_multiple_cards(parent, base_url, source):
            break
        if parent.name in {"article", "li"}:
            return parent
        current = parent
    return current


def extract_article_metadata(
    html_text: str,
    url: str,
) -> tuple[str | None, str | None, str | None]:
    """Extract (title, published_at, summary) from an article HTML page."""
    title, published_at, summary, _excerpt = extract_article_details(html_text, url)
    return title, published_at, summary


def extract_article_details(
    html_text: str,
    url: str,
) -> tuple[str | None, str | None, str | None, str]:
    """Extract (title, published_at, summary, excerpt) from an article HTML page."""
    soup = BeautifulSoup(html_text, "html.parser")
    title: str | None = None
    published_at: str | None = extract_date_from_url(url)
    meta_summary: str | None = None

    # 1. JSON-LD structured data (TechArticle / BlogPosting / Article / NewsArticle)
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw_json = script.string or script.get_text()
        if not raw_json:
            continue
        for obj in _iter_json_ld_objects(raw_json):
            obj_type = obj.get("@type", "")
            types = obj_type if isinstance(obj_type, list) else [obj_type]
            if any(t in {"Article", "BlogPosting", "TechArticle", "NewsArticle"} for t in types):
                title = title or _opt_clean(obj.get("headline") or obj.get("name"), 200)
                published_at = published_at or parse_date(str(obj.get("datePublished") or ""))
                meta_summary = meta_summary or _opt_clean(obj.get("description"), SUMMARY_MAX_LEN)

    # 2. Standard meta tags (OpenGraph, article:published_time, schema.org itemprop)
    meta_title = _meta_content(soup, property="og:title") or _meta_content(soup, name="twitter:title")
    title = title or (_strip_site_suffix(meta_title) if meta_title else None)
    if not published_at:
        for attrs in (
            {"property": "article:published_time"},
            {"itemprop": "datePublished"},
            {"name": "publish_date"},
            {"name": "pubdate"},
            {"name": "date"},
        ):
            candidate = _meta_content(soup, **attrs)
            parsed = parse_date(candidate)
            if parsed:
                published_at = parsed
                break

    if not meta_summary:
        meta_summary = (
            _meta_content(soup, property="og:description", max_len=SUMMARY_MAX_LEN)
            or _meta_content(soup, name="description", max_len=SUMMARY_MAX_LEN)
            or _meta_content(soup, name="twitter:description", max_len=SUMMARY_MAX_LEN)
            or _meta_content(soup, property="twitter:description", max_len=SUMMARY_MAX_LEN)
        )

    # 3. <time>, <h1> / <title>, and structured body extraction
    if not published_at:
        for time_tag in soup.find_all("time"):
            published_at = parse_date(str(time_tag.get("datetime") or "")) or parse_date(
                time_tag.get_text(" ", strip=True)
            )
            if published_at:
                break

    if not title:
        h1 = soup.find("h1")
        if h1:
            title = clean_text(h1.get_text(" ", strip=True), max_len=200)
    if not title and soup.title:
        title = _strip_site_suffix(clean_text(soup.title.get_text(" ", strip=True), max_len=200))

    lead_paragraph, body_excerpt = _extract_structured_body(_select_content_root(soup), title)
    if meta_summary and not _is_stub_text(meta_summary, title, min_len=80):
        summary = meta_summary
    else:
        summary = lead_paragraph or meta_summary
    excerpt = body_excerpt or (summary or "")
    return title, published_at, summary, excerpt


def _is_stub_text(text: str | None, title: str | None = None, min_len: int = STUB_SUMMARY_MAX_LEN) -> bool:
    if not text:
        return True
    cleaned = text.strip()
    if len(cleaned) < min_len:
        return True
    if title:
        low_t = title.strip().lower()
        low_c = cleaned.lower()
        if low_c == low_t or low_t.endswith(low_c) or low_c in low_t:
            return True
    return False


def _needs_article_enrichment(
    title: str | None,
    published_at: str | None,
    summary: str,
    excerpt: str,
    has_heading: bool,
) -> bool:
    if not has_heading or not title or not published_at:
        return True
    if _is_stub_text(summary, title, min_len=STUB_SUMMARY_MAX_LEN):
        return True
    return len(excerpt.strip()) < MIN_ENRICHED_EXCERPT_LEN


def _select_content_root(soup: BeautifulSoup) -> Tag:
    article = soup.find("article")
    if article and len(clean_text(article.get_text(" ", strip=True), max_len=0)) >= 200:
        return article
    main = soup.find("main")
    if main:
        return main
    return soup.body or soup


def _extract_lead_and_excerpt_from_html(
    raw_html: str,
    title: str | None = None,
) -> tuple[str | None, str]:
    if not raw_html:
        return None, ""
    soup = BeautifulSoup(raw_html, "html.parser")
    root = soup.body or soup
    lead, excerpt = _extract_structured_body(root, title)
    return lead, excerpt or clean_text(raw_html, max_len=EXCERPT_MAX_LEN)


def _strip_hidden_and_math(root: Tag) -> None:
    for bad in root.find_all(["script", "style", "noscript", "svg", "math"]):
        bad.decompose()
    for bad in root.find_all(
        class_=lambda c: bool(c and "katex-mathml" in " ".join(c if isinstance(c, list) else [c]))
    ):
        bad.decompose()


def _is_in_noise_container(tag: Tag, root: Tag) -> bool:
    for parent in tag.parents:
        if parent is root:
            break
        if not isinstance(parent, Tag):
            continue
        if parent.name in SKIP_PROSE_PARENTS:
            return True
        if parent.name == "figure":
            cls = " ".join(parent.get("class") or [])
            # Skip non-richtext <figure> blocks (image captions), while preserving Webflow richtext article bodies.
            if "richtext" not in cls.lower() and len(clean_text(parent.get_text(" ", strip=True), max_len=300)) < 260:
                return True
        cls = " ".join(parent.get("class") or [])
        elem_id = str(parent.get("id") or "")
        if LEAF_NOISE_RE.search(cls) or LEAF_NOISE_RE.search(elem_id):
            return True
    return False


def _clean_heading(raw: str) -> str:
    text = clean_text(raw, max_len=100)
    text = re.sub(r"^Link\s+", "", text)
    text = re.sub(r"\s*[#¶]+$", "", text).strip()
    return text


def _is_chrome_heading(text: str) -> bool:
    low = text.lower().strip(": ")
    if not low or len(low) < 3:
        return True
    if low in CHROME_HEADINGS:
        return True
    if any(sub in low for sub in CHROME_HEADING_SUBSTRS):
        return True
    if low.startswith("explore ") and low.endswith(" today"):
        return True
    return False


def _is_boilerplate_para(text: str, title: str | None) -> bool:
    low = text.lower()
    if len(text) < 45:
        return True
    if title:
        low_t = title.lower()
        if low == low_t or low_t.endswith(low) or low in low_t:
            return True
    if any(phrase in low for phrase in BOILERPLATE_PHRASES):
        return True
    if LEAD_SKIP_RE.search(text):
        return True
    return False


def _extract_structured_body(root: Tag, title: str | None) -> tuple[str | None, str]:
    """Extract (lead_paragraph, structured_excerpt) from an article container or HTML fragment."""
    _strip_hidden_and_math(root)

    headings: list[str] = []
    seen_h: set[str] = set()
    for heading in root.find_all(["h2", "h3"]):
        if _is_in_noise_container(heading, root):
            continue
        htext = _clean_heading(heading.get_text(" ", strip=True))
        if _is_chrome_heading(htext) or htext.lower() in seen_h:
            continue
        seen_h.add(htext.lower())
        headings.append(htext)
        if len(headings) >= 8:
            break

    candidate_paras = [p for p in root.find_all("p") if not _is_in_noise_container(p, root)]
    h1 = root.find("h1")
    if h1 is not None:
        after_h1 = [
            p
            for p in h1.find_all_next("p")
            if root in p.parents and not _is_in_noise_container(p, root)
        ]
        if len(after_h1) >= 2:
            candidate_paras = after_h1

    paras: list[str] = []
    seen_p: set[str] = set()
    for p in candidate_paras:
        ptext = clean_text(p.get_text(" ", strip=True), max_len=600)
        if _is_boilerplate_para(ptext, title) or ptext in seen_p:
            continue
        seen_p.add(ptext)
        paras.append(ptext)
        if sum(len(x) for x in paras) >= EXCERPT_MAX_LEN:
            break

    lead = clean_text(paras[0], max_len=SUMMARY_MAX_LEN) if paras else None
    parts: list[str] = []
    if headings:
        parts.append("Sections: " + " | ".join(headings))
    if paras:
        parts.append("\n".join(paras))
    excerpt = clean_text("\n".join(parts), max_len=EXCERPT_MAX_LEN) if parts else ""
    return lead, excerpt


def _iter_json_ld_objects(raw_json: str) -> list[dict]:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return []
    queue = data if isinstance(data, list) else [data]
    objects: list[dict] = []
    while queue:
        item = queue.pop(0)
        if isinstance(item, dict):
            objects.append(item)
            graph = item.get("@graph")
            if isinstance(graph, list):
                queue.extend(graph)
    return objects


def _meta_content(soup: BeautifulSoup, max_len: int = 200, **attrs: str) -> str | None:
    tag = soup.find("meta", attrs=attrs)
    if not isinstance(tag, Tag):
        return None
    content = tag.get("content")
    return _opt_clean(str(content), max_len) if content else None


def _opt_clean(value: object, max_len: int) -> str | None:
    if not value:
        return None
    cleaned = clean_text(str(value), max_len=max_len)
    return cleaned or None


def _strip_site_suffix(title: str) -> str:
    return re.split(r"\s+[|\-–—\\]\s+(?=[^|\-–—\\]+$)", title, maxsplit=1)[0].strip()


def _slug_to_title(url: str) -> str:
    slug = urllib.parse.urlsplit(url).path.rstrip("/").split("/")[-1]
    slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", slug)
    slug = re.sub(r"\.html?$", "", slug)
    return slug.replace("-", " ").replace("_", " ").strip().title() or url


def fetch_source_posts(
    get_text: Callable[[str], str],
    source: SourceConfig,
    seen_keys: set[str],
) -> list[RawPost]:
    """Fetch and normalize posts for one configured source, enriching only unseen articles."""
    extractor = MODE_EXTRACTORS[source.mode]
    raw_posts = extractor(get_text, source, seen_keys)

    canonical_seen = {canonical_key(k) for k in seen_keys}
    enriched: list[RawPost] = []
    for post in raw_posts:
        key = canonical_key(post.url)
        if post.needs_enrichment and key not in canonical_seen:
            try:
                article_html = get_text(post.url)
                meta_title, meta_date, meta_summary, meta_excerpt = extract_article_details(
                    article_html, post.url
                )
                post.title = meta_title or post.title
                post.published_at = post.published_at or meta_date
                if meta_summary and (
                    not post.summary
                    or _is_stub_text(post.summary, post.title, min_len=STUB_SUMMARY_MAX_LEN)
                    or bool(LEAD_SKIP_RE.search(post.summary))
                ):
                    if (
                        not post.summary
                        or len(meta_summary) > len(post.summary)
                        or bool(LEAD_SKIP_RE.search(post.summary))
                    ):
                        post.summary = meta_summary
                if meta_excerpt and len(meta_excerpt) > len(post.excerpt):
                    post.excerpt = meta_excerpt
            except Exception:
                # Keep card-level fallback data if an individual article fetch fails.
                pass
            post.needs_enrichment = False
        if not post.excerpt:
            post.excerpt = post.summary
        enriched.append(post)
    return enriched


def _extract_rss(
    get_text: Callable[[str], str],
    source: SourceConfig,
    _seen_keys: set[str],
) -> list[RawPost]:
    assert source.feed_url
    return parse_rss_entries(get_text(source.feed_url), source)


def _extract_html(
    get_text: Callable[[str], str],
    source: SourceConfig,
    _seen_keys: set[str],
) -> list[RawPost]:
    assert source.index_url
    return parse_html_links(get_text(source.index_url), source.index_url, source)


def _extract_sitemap(
    get_text: Callable[[str], str],
    source: SourceConfig,
    _seen_keys: set[str],
) -> list[RawPost]:
    assert source.sitemap_url
    entries, child_sitemaps = parse_sitemap_xml(get_text(source.sitemap_url))
    for child_url in child_sitemaps:
        if source.sitemap_pattern and not re.search(source.sitemap_pattern, child_url):
            continue
        sub_entries, _ = parse_sitemap_xml(get_text(child_url))
        entries.extend(sub_entries)

    matched: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for url, lastmod in entries:
        key = canonical_key(url)
        if key in seen or not url_matches_source(url, source):
            continue
        seen.add(key)
        matched.append((url, extract_date_from_url(url) or lastmod))

    # Newest first using URL/lastmod date or URL slug fallback.
    matched.sort(key=lambda item: (item[1] or "", item[0]), reverse=True)
    return [
        RawPost(
            url=url,
            title=_slug_to_title(url),
            published_at=pub_date,
            needs_enrichment=True,
        )
        for url, pub_date in matched[: source.max_items]
    ]


MODE_EXTRACTORS: dict[str, Callable[[Callable[[str], str], SourceConfig, set[str]], list[RawPost]]] = {
    "rss": _extract_rss,
    "html_links": _extract_html,
    "sitemap": _extract_sitemap,
}


def build_http_client(timeout: float = 15.0) -> httpx.Client:
    """Create a configured httpx client with browser-like UA and certifi CA bundle."""
    import certifi

    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        verify=certifi.where(),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
