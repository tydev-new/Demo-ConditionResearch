#!/usr/bin/env python3
"""Build data/corpora/<condition_id>.json for the four quick-pick conditions.

Measles keeps the original CDC + trials + PubMed snapshot (moved off
data/corpus.json). The other three conditions are fetched live.

    python scripts/fetch_sources.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sources  # noqa: E402

LEGACY = ROOT / "data" / "corpus.json"
QUERIES = ["heart failure", "type 2 diabetes", "asthma"]


def _status_from_chunks(chunks: list[dict]) -> list[dict]:
    counts = Counter(chunk["source_name"] for chunk in chunks)
    rows = []
    for name, count in counts.items():
        rows.append({"source_name": name, "ok": count > 0, "chunks": count, "error": None})
    return rows


def migrate_measles() -> None:
    destination = sources.corpus_path("measles")
    if destination.is_file():
        print(f"measles corpus already at {destination}")
        return
    if not LEGACY.is_file():
        raise SystemExit(f"Missing {LEGACY} and {destination}; cannot build the measles corpus")
    chunks = json.loads(LEGACY.read_text(encoding="utf-8"))
    for chunk in chunks:
        chunk["condition_id"] = "measles"
    document = {
        "condition": {
            "id": "measles",
            "name": "Measles",
            "synonyms": ["Rubeola"],
            "mesh_id": "D008457",
            "last_fetched": "2026-11-20",
        },
        "chunks": chunks,
        "sources": _status_from_chunks(chunks),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LEGACY.unlink()
    print(f"migrated measles: {len(chunks)} chunks -> {destination}")


def fetch_named(query: str) -> None:
    print(f"Resolving {query!r}...")
    resolved = sources.normalize_condition(query)
    if resolved["status"] != "chosen":
        raise SystemExit(f"MeSH did not auto-resolve {query!r}: {resolved}")
    chosen = resolved["chosen"]
    print(f"  {chosen['name']} {chosen['mesh_id']} -> {chosen['id']}")
    document = sources.gather_sources(chosen)
    path = sources.corpus_path(chosen["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {len(document['chunks'])} chunks")
    for row in document["sources"]:
        print(f"    {row['source_name']}: {row['chunks']} ok={row['ok']} {row['error'] or ''}")


def main() -> None:
    migrate_measles()
    for query in QUERIES:
        fetch_named(query)


if __name__ == "__main__":
    main()
