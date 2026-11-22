"""Export newly archived posts not yet present in backend/data/labels.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..extractors import canonical_key

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export unlabeled posts from data/posts.json for review and dataset expansion."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=REPO_ROOT / "data" / "posts.json",
        help="Path to posts.json archive",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=REPO_ROOT / "backend" / "data" / "labels.jsonl",
        help="Path to existing labeled JSONL dataset",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "backend" / "data" / "unlabeled.jsonl",
        help="Output JSONL path for unlabeled posts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = json.loads(args.archive.read_text(encoding="utf-8"))
    seen_urls: set[str] = set()
    if args.labels.exists():
        for line in args.labels.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("url"):
                seen_urls.add(canonical_key(row["url"]))

    unlabeled = []
    for post in archive.get("posts", []):
        url = str(post.get("url", "")).strip()
        if not url or canonical_key(url) in seen_urls:
            continue
        unlabeled.append(
            {
                "url": url,
                "source_id": post.get("source_id", ""),
                "source_name": post.get("source_name", ""),
                "title": post.get("title", ""),
                "summary": post.get("summary", ""),
                "excerpt": post.get("excerpt", ""),
                "label": "",
                "topics": [],
                "note": "",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in unlabeled) + ("\n" if unlabeled else ""),
        encoding="utf-8",
    )
    print(f"Exported {len(unlabeled)} unlabeled posts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
