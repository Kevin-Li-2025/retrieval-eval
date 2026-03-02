"""
Retrieval metrics: Recall@k, MRR, and extensible registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any

import numpy as np


def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    """Recall@k: proportion of expected docs that appear in top-k retrieved."""
    if not expected_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for eid in expected_ids if eid in top_k)
    return hits / len(expected_ids)


def mrr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant doc (0 if none)."""
    if not expected_ids:
        return 0.0
    expected_set = set(expected_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_set:
            return 1.0 / rank
    return 0.0


@dataclass
class QueryResult:
    query_id: str
    query: str
    expected_ids: List[str]
    retrieved_ids: List[str]
    scores: List[float] = field(default_factory=list)

    @property
    def recall_at_1(self) -> float:
        return recall_at_k(self.retrieved_ids, self.expected_ids, 1)

    @property
    def recall_at_5(self) -> float:
        return recall_at_k(self.retrieved_ids, self.expected_ids, 5)

    @property
    def recall_at_10(self) -> float:
        return recall_at_k(self.retrieved_ids, self.expected_ids, 10)

    @property
    def mrr_score(self) -> float:
        return mrr(self.retrieved_ids, self.expected_ids)

    def is_success(self, k: int = 1) -> bool:
        """True if at least one expected doc is in top-k."""
        return recall_at_k(self.retrieved_ids, self.expected_ids, k) > 0


def compute_metrics(
    results: List[QueryResult],
    k_values: List[int] | None = None,
    include_mrr: bool = True,
) -> Dict[str, float]:
    """Aggregate metrics over a list of query results."""
    k_values = k_values or [1, 5, 10]
    metrics: Dict[str, float] = {}

    for k in k_values:
        recalls = [recall_at_k(r.retrieved_ids, r.expected_ids, k) for r in results]
        metrics[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0

    if include_mrr:
        mrr_scores = [r.mrr_score for r in results]
        metrics["mrr"] = float(np.mean(mrr_scores)) if mrr_scores else 0.0

    metrics["num_queries"] = float(len(results))
    return metrics


def get_failure_cases(
    results: List[QueryResult],
    k: int = 1,
) -> List[Dict[str, Any]]:
    """Return query results where no expected doc is in top-k (for report)."""
    return [
        {
            "query_id": r.query_id,
            "query": r.query,
            "expected_ids": r.expected_ids,
            "retrieved_top_k": r.retrieved_ids[:k],
        }
        for r in results
        if not r.is_success(k=k)
    ]
