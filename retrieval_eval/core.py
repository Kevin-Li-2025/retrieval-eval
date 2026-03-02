"""
Core eval loop: load golden set, run retrieval, compute metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from .schema import load_config, load_golden_set_from_config, load_corpus, load_queries
from .embeddings import get_embedding_provider, EmbeddingProvider
from .retrieval import Retriever, InMemoryRetriever, build_memory_retriever, QdrantRetriever
from .metrics import QueryResult, compute_metrics, get_failure_cases


@dataclass
class EvalResult:
    """Result of a full eval run."""
    metrics: Dict[str, float]
    query_results: List[QueryResult]
    failure_cases: List[Dict[str, Any]]
    config: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "failure_cases": self.failure_cases,
            "num_queries": len(self.query_results),
            "config": self.config,
        }


def load_golden_set(
    corpus_path: str | Path,
    queries_path: str | Path,
) -> tuple[List[Dict], List[Dict]]:
    """Load corpus and queries from paths. Returns (corpus, queries)."""
    from .schema import load_corpus, load_queries
    corpus = load_corpus(Path(corpus_path))
    queries = load_queries(Path(queries_path))
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


def run_eval(
    corpus: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    embedding_provider: str | EmbeddingProvider = "openai",
    retriever: str | Retriever = "memory",
    top_k: int = 10,
    k_values: Optional[List[int]] = None,
    include_mrr: bool = True,
    embedding_kwargs: Optional[Dict] = None,
    retriever_kwargs: Optional[Dict] = None,
) -> EvalResult:
    """
    Run full evaluation: embed corpus (if memory), embed queries, search, aggregate metrics.
    """
    embedding_kwargs = embedding_kwargs or {}
    retriever_kwargs = retriever_kwargs or {}
    k_values = k_values or [1, 5, 10]

    if isinstance(embedding_provider, str):
        provider = get_embedding_provider(embedding_provider, **embedding_kwargs)
    else:
        provider = embedding_provider

    doc_ids = [c["id"] for c in corpus]
    texts = [c["text"] for c in corpus]

    if retriever == "memory" or isinstance(retriever, InMemoryRetriever):
        if isinstance(retriever, InMemoryRetriever):
            retriever_impl = retriever
        else:
            vectors = provider.embed(texts)
            retriever_impl = InMemoryRetriever(doc_ids, vectors)
    elif retriever == "qdrant":
        retriever_impl = QdrantRetriever(**retriever_kwargs)
    else:
        retriever_impl = retriever

    query_texts = [q["query"] for q in queries]
    query_vectors = provider.embed(query_texts)

    query_results: List[QueryResult] = []
    for i, q in enumerate(queries):
        qvec = query_vectors[i]
        ids, scores = retriever_impl.search(qvec, top_k=top_k)
        query_id = q.get("query_id", str(i))
        expected = q.get("expected_ids") or q.get("expected_doc_ids") or []
        if isinstance(expected, str):
            expected = [expected]
        query_results.append(
            QueryResult(
                query_id=query_id,
                query=q["query"],
                expected_ids=expected,
                retrieved_ids=ids,
                scores=scores,
            )
        )

    metrics = compute_metrics(query_results, k_values=k_values, include_mrr=include_mrr)
    failure_cases = get_failure_cases(query_results, k=1)

    return EvalResult(
        metrics=metrics,
        query_results=query_results,
        failure_cases=failure_cases,
        config={
            "embedding": embedding_provider if isinstance(embedding_provider, str) else "custom",
            "retriever": retriever if isinstance(retriever, str) else "custom",
            "top_k": top_k,
        },
        top_k=top_k,
    )


def run_eval_from_config(
    config_path: str | Path,
    embedding: str = "openai",
    retriever: str = "memory",
    output_path: Optional[str | Path] = None,
    **kwargs,
) -> EvalResult:
    """Load config (YAML), run eval, optionally write JSON report."""
    config_path = Path(config_path)
    config_dir = config_path.parent
    config = load_config(config_path)
    corpus, queries = load_golden_set_from_config(config, config_dir)
    metrics_cfg = config.get("metrics") or {}
    k_values = metrics_cfg.get("k", [1, 5, 10])
    include_mrr = metrics_cfg.get("mrr", True)
    result = run_eval(
        corpus,
        queries,
        embedding_provider=embedding,
        retriever=retriever,
        top_k=max(k_values) if k_values else 10,
        k_values=k_values,
        include_mrr=include_mrr,
        **kwargs,
    )
    result.config.update({"config_file": str(config_path), **config})
    if output_path:
        import json
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    return result
