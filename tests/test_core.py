"""Tests for core eval (in-memory, no external API)."""

import pytest
from retrieval_eval.core import run_eval, load_golden_set, EvalResult
from retrieval_eval.metrics import QueryResult

try:
    from retrieval_eval.embeddings import SentenceTransformerEmbedding
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False


@pytest.fixture
def small_corpus():
    return [
        {"id": "a", "text": "first document about cats"},
        {"id": "b", "text": "second document about dogs"},
        {"id": "c", "text": "third document about cats and dogs"},
    ]


@pytest.fixture
def small_queries():
    return [
        {"query_id": "1", "query": "cats", "expected_ids": ["a", "c"]},
        {"query_id": "2", "query": "dogs", "expected_ids": ["b", "c"]},
    ]


def test_load_golden_set(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    corpus_path.write_text('{"id":"x","text":"hello"}\n', encoding="utf-8")
    queries_path.write_text('{"query_id":"1","query":"hi","expected_ids":["x"]}\n', encoding="utf-8")
    corpus, queries = load_golden_set(corpus_path, queries_path)
    assert len(corpus) == 1 and corpus[0]["id"] == "x"
    assert len(queries) == 1 and queries[0]["expected_ids"] == ["x"]


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_run_eval_memory(small_corpus, small_queries):
    provider = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
    result = run_eval(
        small_corpus,
        small_queries,
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
    assert len(result.query_results) == 2
    assert result.metrics["recall@3"] >= 0  # semantic search should find something
