"""CLI entrypoint to refresh blog sources and export static JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..updater import run_update

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch AI/LLM infra blogs, update persistent archive, and export static JSON."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "sources.yaml",
        help="Path to sources.yaml config",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=REPO_ROOT / "data" / "posts.json",
        help="Path to persistent posts.json archive",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=REPO_ROOT / "frontend" / "public" / "data",
        help="Optional directory to write static posts.json for the Vite frontend",
    )
    parser.add_argument(
        "--from-archive",
        action="store_true",
        help="Re-export static JSON from data/posts.json without network fetches",
    )
    parser.add_argument(
        "--refresh-thin-excerpts",
        action="store_true",
        help="Re-enrich archived posts whose cached excerpt is still thin or boilerplate",
    )
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help="Re-score all archived posts with the committed classifier instead of new posts only",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Max concurrent source fetch workers",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_update(
        config_path=args.config,
        archive_path=args.archive,
        export_dir=args.export_dir,
        from_archive=args.from_archive,
        refresh_thin_excerpts=args.refresh_thin_excerpts,
        reclassify=args.reclassify,
        max_workers=args.workers,
    )
    ok_sources = sum(1 for s in payload["sources"] if s["status"] == "ok")
    err_sources = [s for s in payload["sources"] if s["status"] != "ok"]
    print(
        f"Updated {len(payload['posts'])} archived posts "
        f"(+{payload['last_new_count']} new) across {ok_sources}/{len(payload['sources'])} sources."
    )
    for err in err_sources:
        print(f"  WARN [{err['id']}]: {err['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
