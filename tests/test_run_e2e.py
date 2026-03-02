"""End-to-end test: run_eval with a dummy in-memory embedder (no OpenAI/sentence-transformers)."""

import numpy as np
import pytest
from retrieval_eval.core import run_eval, EvalResult
from retrieval_eval.embeddings import EmbeddingProvider


class DummyEmbedding(EmbeddingProvider):
    """Deterministic dummy: same text -> same vector (hash-based)."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    def embed(self, texts):
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hash(t) % (2 ** 31)
            np.random.seed(h)
            out[i] = np.random.randn(self._dim).astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    @property
    def dimension(self):
        return self._dim


def test_run_eval_with_dummy_embedding():
    corpus = [
        {"id": "d1", "text": "python programming"},
        {"id": "d2", "text": "machine learning"},
        {"id": "d3", "text": "python and machine learning"},
    ]
    queries = [
        {"query_id": "q1", "query": "python programming", "expected_ids": ["d1", "d3"]},
        {"query_id": "q2", "query": "machine learning", "expected_ids": ["d2", "d3"]},
    ]
    provider = DummyEmbedding(dim=4)
    result = run_eval(
        corpus,
        queries,
        embedding_provider=provider,
        retriever="memory",
        top_k=3,
        k_values=[1, 3],
        include_mrr=True,
    )
    assert isinstance(result, EvalResult)
    assert "recall@1" in result.metrics
    assert "recall@3" in result.metrics
    assert "mrr" in result.metrics
    assert result.metrics["num_queries"] == 2
    assert len(result.query_results) == 2
    assert 0 <= result.metrics["recall@1"] <= 1
    assert 0 <= result.metrics["recall@3"] <= 1
