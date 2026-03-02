"""
Golden set and config schema: corpus + queries with expected doc ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import yaml


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL: one JSON object per line."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_json(path: Path) -> List[Dict[str, Any]]:
    """Load JSON: expect list of objects or single object (wrapped in list)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def load_corpus(path: Path) -> List[Dict[str, Any]]:
    """Corpus: list of {id, text} (or id + content)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    return load_json(path)


def load_queries(path: Path) -> List[Dict[str, Any]]:
    """Queries: list of {query_id, query, expected_ids} (expected_ids = list of doc ids)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    return load_json(path)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config for eval suite."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_golden_set_from_config(config: Dict[str, Any], config_dir: Path) -> tuple:
    """
    Returns (corpus_list, queries_list).
    corpus_list: [{"id": ..., "text": ...}, ...]
    queries_list: [{"query_id": ..., "query": ..., "expected_ids": [...]}, ...]
    """
    corpus_path = config_dir / config["corpus"]["path"]
    queries_path = config_dir / config["queries"]["path"]
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    # Normalize keys
    for c in corpus:
        if "text" not in c and "content" in c:
            c["text"] = c["content"]
        if "id" not in c and "doc_id" in c:
            c["id"] = c["doc_id"]
    for q in queries:
        if "expected_ids" not in q and "expected_doc_ids" in q:
            q["expected_ids"] = q["expected_doc_ids"]
        if "expected_ids" in q and isinstance(q["expected_ids"], str):
            q["expected_ids"] = [q["expected_ids"]]
    return corpus, queries
