"""Unit tests for RSS, sitemap, HTML link extraction, and date/URL normalization."""

from __future__ import annotations

import unittest

from app.extractors import (
    canonical_key,
    extract_article_details,
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

        # First confirm direct HTML parsing flags thin card excerpts and missing metadata for enrichment.
        direct = parse_html_links(index_html, "https://humansand.ai/blogs/", src)
        self.assertEqual(len(direct), 2)
        self.assertTrue(direct[0].needs_enrichment)
        self.assertEqual(direct[0].published_at, "2026-11-10T00:00:00Z")
        self.assertTrue(direct[1].needs_enrichment)

        # Verify seen URLs are skipped for enrichment, while unseen URLs are enriched once.
        fetches: list[str] = []

        def fake_get(url: str) -> str:
            fetches.append(url)
            if url == "https://humansand.ai/blogs/":
                return index_html
            if url == "https://humansand.ai/blog/gpu-weight-teleport":
                return """<html><article><h1>GPU Weight Teleportation</h1>
                  <h2>Delta update algebra</h2>
                  <p>Lossless GPU compression for instant RL weight sync across Blackwell nodes.</p>
                </article></html>"""
            if url == "https://humansand.ai/blog/nvfp4-rl":
                return article_html
            raise AssertionError(f"Unexpected fetch: {url}")

        posts = fetch_source_posts(fake_get, src, seen_keys=set())
        self.assertEqual(posts[1].published_at, "2026-10-01T00:00:00Z")
        self.assertEqual(posts[1].summary, "Balancing stability in NVFP4 RL.")
        self.assertIn("Delta update algebra", posts[0].excerpt)
        self.assertEqual(len(fetches), 3)

        # When already seen, article pages should not be fetched again.
        fetches.clear()
        fetch_source_posts(
            fake_get,
            src,
            seen_keys={
                "https://humansand.ai/blog/gpu-weight-teleport",
                "https://humansand.ai/blog/nvfp4-rl",
            },
        )
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
        self.assertEqual(direct[0].summary, "This featured post has its own full card description.")
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

    def test_headingless_card_enriches_and_strips_og_title_suffix(self) -> None:
        index_html = """
        <html><body>
          <a href="./news/ai-infra">
            <p>AI Infrastructure at Periodic</p>
            <p>Sep 15, 2026</p>
          </a>
        </body></html>"""
        article_html = """
        <html><head>
          <meta property="og:title" content="AI Infrastructure at Periodic – Periodic Labs" />
          <meta property="og:description" content="Training, inference, and sandboxing infrastructure." />
        </head><body><h1>AI Infrastructure at Periodic</h1></body></html>"""
        src = SourceConfig(
            id="periodic",
            name="Periodic Labs",
            homepage="https://periodic.com/news",
            mode="html_links",
            index_url="https://periodic.com/news",
            url_pattern=r"^https://periodic\.com/news/[^/?#]+/?$",
        )

        def fake_get(url: str) -> str:
            if url == "https://periodic.com/news":
                return index_html
            if url == "https://periodic.com/news/ai-infra":
                return article_html
            raise AssertionError(f"Unexpected fetch: {url}")

        posts = fetch_source_posts(fake_get, src, seen_keys=set())
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].title, "AI Infrastructure at Periodic")
        self.assertEqual(posts[0].published_at, "2026-09-15T00:00:00Z")
        self.assertEqual(posts[0].summary, "Training, inference, and sandboxing infrastructure.")

    def test_rich_article_excerpt_headings_boilerplate_and_webflow_figure(self) -> None:
        html_text = """
        <html>
          <head>
            <meta property="og:title" content="The path to Mojo 1.0" />
            <meta property="og:description" content="The path to Mojo 1.0" />
          </head>
          <body>
            <main>
              <p>This website requires JavaScript in order to work properly.</p>
              <h1>The path to Mojo 1.0</h1>
              <figure class="text-rich-text w-richtext">
                <h2>AI-Generated Summary</h2>
                <h2>Link Compiler Architecture #</h2>
                <p>Special thanks to early contributors for feedback and edits on this draft.</p>
                <p>Mojo started with a focus on writing high-performance kernels for GPUs and CPUs across accelerators.<math>x_1</math></p>
                <figure><figcaption><p>Small caption ignored in body prose.</p></figcaption></figure>
                <h2>Related tags</h2>
              </figure>
            </main>
          </body>
        </html>"""
        title, _pub, summary, excerpt = extract_article_details(
            html_text, "https://www.modular.com/blog/the-path-to-mojo-1-0"
        )
        self.assertEqual(title, "The path to Mojo 1.0")
        self.assertEqual(
            summary,
            "Mojo started with a focus on writing high-performance kernels for GPUs and CPUs across accelerators.",
        )
        self.assertIn("Sections: Compiler Architecture", excerpt)
        self.assertNotIn("AI-Generated Summary", excerpt)
        self.assertNotIn("Related tags", excerpt)
        self.assertNotIn("Special thanks", excerpt)
        self.assertNotIn("requires JavaScript", excerpt)
        self.assertNotIn("Small caption ignored", excerpt)

    def test_rss_content_encoded_and_stub_summary_enrichment(self) -> None:
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
          <channel>
            <item>
              <title>Full Feed Post</title>
              <link>https://example.com/blog/full-feed</link>
              <pubDate>Thu, 12 Nov 2026 08:00:00 GMT</pubDate>
              <description>Short RSS blurb that would normally hide the full body.</description>
              <content:encoded><![CDATA[
                <h2>Runtime Internals</h2>
                <p>First detailed technical paragraph explaining CUDA graph capture across worker processes, kernel fusion boundaries, warp-specialized tensor core scheduling, and batch size bucket selection across distributed GPU inference hosts in large multi-node clusters.</p>
                <p>Second detailed paragraph measuring host memory overhead, paged KV-cache fragmentation, prefix tree eviction behavior, and tail scheduler latency reductions across hundreds of inference nodes running mixed prefill and decode workloads under skewed production traffic.</p>
                <p>Third detailed paragraph detailing asynchronous IPC weight synchronization, non-blocking collective communication overlap, and end-to-end token throughput improvements across H100 and Blackwell clusters.</p>
              ]]></content:encoded>
            </item>
            <item>
              <title>Teaser Content Feed Post</title>
              <link>https://example.com/blog/teaser-content</link>
              <pubDate>Thu, 12 Nov 2026 08:30:00 GMT</pubDate>
              <description>Short feed summary for teaser post.</description>
              <content:encoded><![CDATA[<p>This content block is only a short 350-char teaser like some developer blogs emit in their RSS feed instead of the full article body, so the extractor still needs to visit the page.</p>]]></content:encoded>
            </item>
            <item>
              <title>Modular: Stub RSS Post</title>
              <link>https://example.com/blog/stub-feed</link>
              <pubDate>Thu, 12 Nov 2026 09:00:00 GMT</pubDate>
              <description>Stub RSS Post</description>
            </item>
          </channel>
        </rss>"""
        src = SourceConfig(
            id="demo-rss",
            name="Demo RSS",
            homepage="https://example.com/blog",
            mode="rss",
            feed_url="https://example.com/feed.xml",
        )
        fetches: list[str] = []

        def fake_get(url: str) -> str:
            fetches.append(url)
            if url == "https://example.com/feed.xml":
                return rss_xml
            if url == "https://example.com/blog/teaser-content":
                return """<html><article>
                  <h1>Teaser Content Feed Post</h1>
                  <h2>Sparse Attention Design</h2>
                  <p>Full article body explaining sparse attention indexing, top-k KV cache block selection, fused hardware epilogues, and DRAM memory bandwidth gains on long-context inference workloads across modern multi-GPU clusters.</p>
                </article></html>"""
            if url == "https://example.com/blog/stub-feed":
                return """<html><article>
                  <h1>Modular: Stub RSS Post</h1>
                  <h2>Kernel Dispatch</h2>
                  <p>Detailed engineering walkthrough of async tensor core scheduling on modern accelerators.</p>
                </article></html>"""
            raise AssertionError(f"Unexpected fetch: {url}")

        posts = fetch_source_posts(fake_get, src, seen_keys=set())
        self.assertEqual(len(posts), 3)
        # Full content:encoded (>=750 chars) avoids extra fetch, while teaser content:encoded fetches full article.
        self.assertEqual(
            fetches,
            [
                "https://example.com/feed.xml",
                "https://example.com/blog/teaser-content",
                "https://example.com/blog/stub-feed",
            ],
        )
        self.assertIn("Sections: Runtime Internals", posts[0].excerpt)
        self.assertIn("CUDA graph capture", posts[0].excerpt)
        self.assertIn("Sections: Sparse Attention Design", posts[1].excerpt)
        # Stub RSS description is upgraded in both summary and excerpt from fetched page.
        self.assertEqual(
            posts[2].summary,
            "Detailed engineering walkthrough of async tensor core scheduling on modern accelerators.",
        )
        self.assertIn("Sections: Kernel Dispatch", posts[2].excerpt)

    def test_brand_title_byline_skip_and_duplicate_sitewide_summary_recovery(self) -> None:
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://www.cerebras.ai/blog/moe-guide-calculator</loc><lastmod>2026-10-22</lastmod></url>
          <url><loc>https://www.cerebras.ai/blog/wafer-yield</loc><lastmod>2026-10-21</lastmod></url>
        </urlset>"""
        page1 = """<html><head>
          <meta property="og:title" content="Cerebras" />
          <meta property="og:description" content="Generic sitewide marketing slogan repeated across all pages." />
        </head><body><article>
          <h1>MoE Math Demystified</h1>
          <h2>Compute Balance</h2>
          <p>MoE Fundamentals | Router Wars | Debugging | MoE at Scale</p>
          <p>By Ryan Lopopolo, Member of the Technical Staff</p>
          <p>What does 8x7B actually mean when computing active parameter counts and KV-cache memory?</p>
        </article></body></html>"""
        page2 = """<html><head>
          <meta property="og:title" content="Wafer-Scale Chip Yield" />
          <meta property="og:description" content="Generic sitewide marketing slogan repeated across all pages." />
        </head><body><article>
          <h1>Wafer-Scale Chip Yield</h1>
          <h2>Redundancy</h2>
          <p>Originally published October 6th on my personal blog.</p>
          <p>With contributions from Sherif Cherfa and Halley Chang.</p>
          <p>MCP had a formative year. Then it had a turbulent week.</p>
          <p>By contrast, wafer-scale integration builds redundancy directly into the die interconnect fabric.</p>
        </article></body></html>"""
        src = SourceConfig(
            id="cerebras",
            name="Cerebras",
            homepage="https://www.cerebras.ai/blog",
            mode="sitemap",
            sitemap_url="https://www.cerebras.ai/sitemap.xml",
            url_pattern=r"^https://www\.cerebras\.ai/blog/[^/?#]+/?$",
        )

        def fake_get(url: str) -> str:
            if url == "https://www.cerebras.ai/sitemap.xml":
                return sitemap_xml
            if url == "https://www.cerebras.ai/blog/moe-guide-calculator":
                return page1
            if url == "https://www.cerebras.ai/blog/wafer-yield":
                return page2
            raise AssertionError(f"Unexpected fetch: {url}")

        posts = fetch_source_posts(fake_get, src, seen_keys=set())
        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0].title, "MoE Math Demystified")
        self.assertEqual(
            posts[0].summary,
            "What does 8x7B actually mean when computing active parameter counts and KV-cache memory?",
        )
        self.assertNotIn("MoE Fundamentals | Router Wars", posts[0].excerpt)
        self.assertNotIn("By Ryan Lopopolo", posts[0].excerpt)
        self.assertEqual(posts[1].title, "Wafer-Scale Chip Yield")
        self.assertEqual(
            posts[1].summary,
            "MCP had a formative year. Then it had a turbulent week. By contrast, wafer-scale integration builds redundancy directly into the die interconnect fabric.",
        )
        self.assertNotIn("With contributions from", posts[1].excerpt)


if __name__ == "__main__":
    unittest.main()
