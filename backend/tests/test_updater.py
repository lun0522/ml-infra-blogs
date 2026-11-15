"""Unit tests for persistent archive merging and failure isolation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.updater import run_update


SAMPLE_CONFIG = """sources:
  - id: vllm
    name: vLLM Blog
    homepage: https://vllm.ai/blog
    mode: rss
    feed_url: https://vllm.ai/blog/rss.xml
    max_items: 10
  - id: lmsys
    name: LMSYS / SGLang
    homepage: https://www.lmsys.org/blog/
    mode: sitemap
    sitemap_url: https://www.lmsys.org/sitemap.xml
    url_pattern: ^https://www\\.lmsys\\.org/blog/\\d{4}-\\d{2}-\\d{2}-
    max_items: 10
"""


class UpdaterTests(unittest.TestCase):
    def test_persistent_merge_preserves_first_seen_and_handles_source_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "sources.yaml"
            archive = root / "data" / "posts.json"
            export_dir = root / "public" / "data"
            cfg.write_text(SAMPLE_CONFIG, encoding="utf-8")

            responses = {
                "https://vllm.ai/blog/rss.xml": """<?xml version="1.0"?>
                    <rss version="2.0"><channel>
                      <item>
                        <title>Post One</title>
                        <link>https://vllm.ai/blog/post-one</link>
                        <pubDate>Thu, 12 Nov 2026 08:00:00 GMT</pubDate>
                      </item>
                    </channel></rss>""",
                "https://www.lmsys.org/sitemap.xml": """<?xml version="1.0"?>
                    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
                      <url><loc>https://www.lmsys.org/blog/2026-11-13-sglang</loc></url>
                    </urlset>""",
                "https://www.lmsys.org/blog/2026-11-13-sglang": """
                    <html><head><meta property="og:title" content="SGLang Post Title" /></head></html>""",
            }

            first = run_update(
                config_path=cfg,
                archive_path=archive,
                export_dir=export_dir,
                http_get=responses.__getitem__,
            )
            self.assertEqual(first["last_new_count"], 2)
            self.assertEqual(len(first["posts"]), 2)
            self.assertEqual(first["posts"][0]["title"], "SGLang Post Title")
            original_first_seen = first["posts"][0]["first_seen"]

            # Simulate second run: lmsys fails, vllm adds Post Two.
            responses["https://vllm.ai/blog/rss.xml"] = """<?xml version="1.0"?>
                <rss version="2.0"><channel>
                  <item>
                    <title>Post Two</title>
                    <link>https://vllm.ai/blog/post-two</link>
                    <pubDate>Fri, 13 Nov 2026 09:00:00 GMT</pubDate>
                  </item>
                  <item>
                    <title>Post One</title>
                    <link>https://vllm.ai/blog/post-one</link>
                    <pubDate>Thu, 12 Nov 2026 08:00:00 GMT</pubDate>
                  </item>
                </channel></rss>"""
            del responses["https://www.lmsys.org/sitemap.xml"]

            second = run_update(
                config_path=cfg,
                archive_path=archive,
                export_dir=export_dir,
                http_get=responses.__getitem__,
            )
            self.assertEqual(second["last_new_count"], 1)
            self.assertEqual(len(second["posts"]), 3)
            lmsys_post = next(p for p in second["posts"] if p["source_id"] == "lmsys")
            self.assertEqual(lmsys_post["title"], "SGLang Post Title")
            self.assertEqual(lmsys_post["first_seen"], original_first_seen)
            lmsys_health = next(s for s in second["sources"] if s["id"] == "lmsys")
            self.assertEqual(lmsys_health["status"], "error")
            self.assertEqual(lmsys_health["total_count"], 1)

            # Verify exported JSON archive is valid.
            exported = json.loads((export_dir / "posts.json").read_text(encoding="utf-8"))
            self.assertEqual(len(exported["posts"]), 3)

            # Verify --from-archive rebuilds cleanly without network access.
            rebuilt = run_update(
                config_path=cfg,
                archive_path=archive,
                export_dir=export_dir,
                from_archive=True,
            )
            self.assertEqual(len(rebuilt["posts"]), 3)

            # Verify archive retains at most each source's max_items newest posts.
            cfg.write_text(
                SAMPLE_CONFIG.replace("max_items: 10", "max_items: 1", 1),
                encoding="utf-8",
            )
            capped = run_update(
                config_path=cfg,
                archive_path=archive,
                export_dir=export_dir,
                from_archive=True,
            )
            vllm_posts = [p for p in capped["posts"] if p["source_id"] == "vllm"]
            self.assertEqual(len(vllm_posts), 1)
            self.assertEqual(vllm_posts[0]["title"], "Post Two")


if __name__ == "__main__":
    unittest.main()
