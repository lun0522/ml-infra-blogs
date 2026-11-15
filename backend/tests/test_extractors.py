"""Unit tests for RSS, sitemap, HTML link extraction, and date/URL normalization."""

from __future__ import annotations

import unittest

from app.extractors import (
    canonical_key,
    extract_article_metadata,
    extract_date_from_url,
    fetch_source_posts,
    normalize_url,
    parse_date,
    parse_html_links,
    parse_rss_entries,
    parse_sitemap_xml,
)
from app.models import SourceConfig


class ExtractorTests(unittest.TestCase):
    def test_url_normalization_and_dedup_key(self) -> None:
        u1 = normalize_url("https://vLLM.ai/blog/paged-attention/?utm_source=x&ref=hn#top")
        u2 = normalize_url("https://vllm.ai/blog/paged-attention")
        self.assertEqual(u1, "https://vllm.ai/blog/paged-attention/")
        self.assertEqual(canonical_key(u1), canonical_key(u2))

    def test_date_parsing_variants(self) -> None:
        self.assertEqual(parse_date("2026-11-05T00:00:00Z"), "2026-11-05T00:00:00Z")
        self.assertEqual(parse_date("Nov 5, 2026"), "2026-11-05T00:00:00Z")
        self.assertEqual(parse_date("20 OCT, 2026"), "2026-10-20T00:00:00Z")
        self.assertEqual(parse_date("November 13, 2026"), "2026-11-13T00:00:00Z")
        self.assertEqual(parse_date("Thu, 12 Nov 2026 10:30:00 GMT"), "2026-11-12T10:30:00Z")
        self.assertEqual(
            extract_date_from_url("https://www.lmsys.org/blog/2026-11-13-miles-vera-sandboxes"),
            "2026-11-13T00:00:00Z",
        )

    def test_parse_rss_entries_filters_and_cleans_html(self) -> None:
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Test Blog</title>
            <item>
              <title>PagedAttention &amp; Beyond</title>
              <link>https://example.com/blog/paged-attention?utm_source=rss</link>
              <pubDate>Thu, 12 Nov 2026 08:00:00 GMT</pubDate>
              <description><![CDATA[<p>Deep dive into <b>KV cache</b> paging.</p>]]></description>
              <author>vLLM Team</author>
            </item>
            <item>
              <title>Careers</title>
              <link>https://example.com/careers</link>
              <pubDate>Thu, 12 Nov 2026 08:00:00 GMT</pubDate>
            </item>
          </channel>
        </rss>"""
        src = SourceConfig(
            id="vllm",
            name="vLLM Blog",
            homepage="https://example.com/blog",
            mode="rss",
            feed_url="https://example.com/feed.xml",
            url_pattern=r"^https://example\.com/blog/",
        )
        posts = parse_rss_entries(rss_xml, src)
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].url, "https://example.com/blog/paged-attention")
        self.assertEqual(posts[0].title, "PagedAttention & Beyond")
        self.assertEqual(posts[0].published_at, "2026-11-12T08:00:00Z")
        self.assertEqual(posts[0].summary, "Deep dive into KV cache paging.")

    def test_parse_html_links_and_lazy_enrichment(self) -> None:
        index_html = """
        <html><body>
          <nav><a href="/blogs/">Blogs</a></nav>
          <ul class="posts">
            <li>
              <a class="post" href="/blog/gpu-weight-teleport">
                <div class="post-copy">
                  <h2>GPU Weight Teleportation</h2>
                  <p>Lossless GPU compression for instant RL weight sync.</p>
                </div>
                <time>Nov 10, 2026</time>
              </a>
            </li>
            <li>
              <a class="post" href="/blog/nvfp4-rl">The 4-bitter Lesson</a>
            </li>
          </ul>
        </body></html>"""
        article_html = """
        <html><head>
          <meta property="og:title" content="The 4-bitter Lesson" />
          <meta property="og:description" content="Balancing stability in NVFP4 RL." />
          <script type="application/ld+json">
            {"@context": "https://schema.org", "@type": "TechArticle", "datePublished": "2026-10-01"}
          </script>
        </head></html>"""
        src = SourceConfig(
            id="humansand",
            name="humans& Blog",
            homepage="https://humansand.ai/blogs/",
            mode="html_links",
            index_url="https://humansand.ai/blogs/",
            url_pattern=r"^https://humansand\.ai/blog/[^/?#]+/?$",
        )

        # First confirm direct HTML parsing flags missing date on second post.
        direct = parse_html_links(index_html, "https://humansand.ai/blogs/", src)
        self.assertEqual(len(direct), 2)
        self.assertFalse(direct[0].needs_enrichment)
        self.assertEqual(direct[0].published_at, "2026-11-10T00:00:00Z")
        self.assertTrue(direct[1].needs_enrichment)

        # Verify seen URLs are skipped for enrichment, while unseen URLs are enriched once.
        fetches: list[str] = []

        def fake_get(url: str) -> str:
            fetches.append(url)
            if url == "https://humansand.ai/blogs/":
                return index_html
            if url == "https://humansand.ai/blog/nvfp4-rl":
                return article_html
            raise AssertionError(f"Unexpected fetch: {url}")

        posts = fetch_source_posts(fake_get, src, seen_keys=set())
        self.assertEqual(posts[1].published_at, "2026-10-01T00:00:00Z")
        self.assertEqual(posts[1].summary, "Balancing stability in NVFP4 RL.")
        self.assertEqual(len(fetches), 2)

        # When already seen, article page should not be fetched again.
        fetches.clear()
        fetch_source_posts(fake_get, src, seen_keys={"https://humansand.ai/blog/nvfp4-rl"})
        self.assertEqual(fetches, ["https://humansand.ai/blogs/"])

    def test_parse_sitemap_and_extract_article_metadata(self) -> None:
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://www.lmsys.org/about</loc></url>
          <url><loc>https://www.lmsys.org/blog/2026-11-13-miles-vera-sandboxes</loc></url>
          <url><loc>https://www.lmsys.org/blog/2026-11-11-unified-memory</loc></url>
        </urlset>"""
        urls, children = parse_sitemap_xml(sitemap_xml)
        self.assertEqual(children, [])
        self.assertEqual(len(urls), 3)

        title, pub, summary = extract_article_metadata(
            """<html><head>
              <meta property="og:title" content="Unified Memory in SGLang" />
              <meta property="article:published_time" content="November 11, 2026" />
              <meta name="description" content="Zero-copy KV paging." />
            </head></html>""",
            "https://www.lmsys.org/blog/2026-11-11-unified-memory",
        )
        self.assertEqual(title, "Unified Memory in SGLang")
        self.assertEqual(pub, "2026-11-11T00:00:00Z")
        self.assertEqual(summary, "Zero-copy KV paging.")

    def test_missing_summary_enrichment_and_card_isolation(self) -> None:
        index_html = """
        <html><body>
          <div class="grid">
            <div class="card">
              <a href="/blog/featured">
                <h2>Featured Post</h2>
                <p>This featured post has its own full card description.</p>
                <time>Nov 12, 2026</time>
              </a>
            </div>
            <div class="card">
              <a href="/blog/compact">
                <h2>Compact Card Post</h2>
                <time>Nov 11, 2026</time>
              </a>
            </div>
          </div>
        </body></html>"""
        compact_html = """
        <html><body><article>
          <h1>Compact Card Post</h1>
          <figure><figcaption><p>Figure caption that should be skipped from summary extraction.</p></figcaption></figure>
          <p>Short note.</p>
          <p>Training low-precision models requires careful scaling so gradients stay accurate across layers.</p>
        </article></body></html>"""
        src = SourceConfig(
            id="demo",
            name="Demo Blog",
            homepage="https://example.com/blog",
            mode="html_links",
            index_url="https://example.com/blog",
            url_pattern=r"^https://example\.com/blog/[^/?#]+/?$",
        )
        direct = parse_html_links(index_html, "https://example.com/blog", src)
        self.assertEqual(len(direct), 2)
        self.assertFalse(direct[0].needs_enrichment)
        self.assertEqual(direct[1].summary, "")
        self.assertTrue(direct[1].needs_enrichment)

        rss_xml = """<?xml version="1.0"?>
        <rss version="2.0"><channel>
          <item>
            <title>No Summary Feed Item</title>
            <link>https://example.com/blog/compact</link>
            <pubDate>Wed, 11 Nov 2026 08:00:00 GMT</pubDate>
          </item>
        </rss>"""
        rss_src = SourceConfig(
            id="demo-rss",
            name="Demo RSS",
            homepage="https://example.com/blog",
            mode="rss",
            feed_url="https://example.com/feed.xml",
        )
        rss_direct = parse_rss_entries(rss_xml, rss_src)
        self.assertEqual(len(rss_direct), 1)
        self.assertTrue(rss_direct[0].needs_enrichment)

        def fake_get(url: str) -> str:
            if url == "https://example.com/feed.xml":
                return rss_xml
            if url == "https://example.com/blog/compact":
                return compact_html
            raise AssertionError(f"Unexpected fetch: {url}")

        rss_enriched = fetch_source_posts(fake_get, rss_src, seen_keys=set())
        self.assertEqual(
            rss_enriched[0].summary,
            "Training low-precision models requires careful scaling so gradients stay accurate across layers.",
        )


if __name__ == "__main__":
    unittest.main()
