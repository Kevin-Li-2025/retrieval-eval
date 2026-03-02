"""
Retrievers: in-memory (exact search) and Qdrant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class Retriever(ABC):
    """Abstract retriever: returns (doc_ids, scores) for a query vector."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[str], List[float]]:
        pass


class InMemoryRetriever(Retriever):
    """Build index from (id, text); embed corpus with given provider and use cosine similarity."""

    def __init__(self, doc_ids: List[str], vectors: np.ndarray):
        assert len(doc_ids) == len(vectors)
        self._ids = doc_ids
        self._vectors = np.asarray(vectors, dtype=np.float32)
        # normalize for cosine
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = self._vectors / norms

    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[str], List[float]]:
        q = np.asarray(query_vector, dtype=np.float32).flatten()
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self._vectors @ q
        order = np.argsort(-scores)[:top_k]
        return [self._ids[i] for i in order], [float(scores[i]) for i in order]


class QdrantRetriever(Retriever):
    """Search an existing Qdrant collection."""

    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
    ):
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("Install qdrant-client: pip install qdrant-client")
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection = collection_name

    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[str], List[float]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        q = query_vector.flatten().tolist()
        results = self._client.search(
            collection_name=self._collection,
            query_vector=q,
            limit=top_k,
            with_payload=True,
        )
        ids = []
        scores = []
        for r in results:
            # Qdrant uses payload or id; assume payload has "id" or use point id
            doc_id = getattr(r.payload, "id", None) or getattr(r, "id", None)
            if doc_id is None and r.payload:
                doc_id = r.payload.get("id")
            ids.append(str(doc_id) if doc_id is not None else str(r.id))
            scores.append(float(r.score))
        return ids, scores


def build_memory_retriever(
    doc_ids: List[str],
    texts: List[str],
    embed_fn,
) -> InMemoryRetriever:
    """Build in-memory index from corpus using provided embed function (e.g. provider.embed)."""
    vectors = embed_fn(texts)
    return InMemoryRetriever(doc_ids, vectors)
