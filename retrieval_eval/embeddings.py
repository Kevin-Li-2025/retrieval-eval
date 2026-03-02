"""
Embedding providers: OpenAI and optional sentence-transformers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return (n, dim) float32 array."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI text-embedding-3-small (or configurable)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dim = 1536 if "small" in model else 3072

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        response = self._client.embeddings.create(model=self._model, input=texts)
        order = sorted(response.data, key=lambda d: d.index)
        return np.array([d.embedding for d in order], dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Local sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install '.[local]'")
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    @property
    def dimension(self) -> int:
        return self._dim


def get_embedding_provider(
    name: str,
    model: str | None = None,
    **kwargs,
) -> EmbeddingProvider:
    """Factory: 'openai' | 'sentence-transformers'."""
    if name == "openai":
        return OpenAIEmbedding(model=model or "text-embedding-3-small", **kwargs)
    if name == "sentence-transformers":
        return SentenceTransformerEmbedding(model_name=model or "all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedding provider: {name}")
