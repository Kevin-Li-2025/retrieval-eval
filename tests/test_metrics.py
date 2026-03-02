"""Tests for retrieval metrics."""

import pytest
from retrieval_eval.metrics import (
    recall_at_k,
    mrr,
    compute_metrics,
    QueryResult,
    get_failure_cases,
)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a"], 1) == 1.0
    assert recall_at_k(["a", "b", "c"], ["a"], 5) == 1.0
    assert recall_at_k(["x", "b", "a"], ["a"], 3) == 1.0
    assert recall_at_k(["x", "y"], ["a"], 2) == 0.0
    assert recall_at_k(["a", "b"], ["a", "b", "c"], 2) == 2 / 3


def test_mrr():
    assert mrr(["a", "b"], ["a"]) == 1.0
    assert mrr(["x", "a"], ["a"]) == 0.5
    assert mrr(["x", "y"], ["a"]) == 0.0
    assert mrr([], ["a"]) == 0.0


def test_compute_metrics():
    results = [
        QueryResult("q1", "query1", ["d1"], ["d1", "d2"], []),
        QueryResult("q2", "query2", ["d2"], ["d2", "d1"], []),
        QueryResult("q3", "query3", ["d1"], ["d3", "d1"], []),
    ]
    m = compute_metrics(results, k_values=[1, 2], include_mrr=True)
    assert m["recall@1"] == pytest.approx(2 / 3)  # q1 ok, q2 ok, q3 miss
    assert m["recall@2"] == 1.0
    assert m["mrr"] == pytest.approx((1 + 1 + 0.5) / 3)
    assert m["num_queries"] == 3


def test_failure_cases():
    results = [
        QueryResult("q1", "q", ["d1"], ["d1"], []),
        QueryResult("q2", "q", ["d2"], ["d3", "d4"], []),
    ]
    failures = get_failure_cases(results, k=1)
    assert len(failures) == 1
    assert failures[0]["query_id"] == "q2"
