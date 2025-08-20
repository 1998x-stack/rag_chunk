# -*- coding: utf-8 -*-
"""Index abstraction with FAISS fallback to NumPy brute-force."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from loguru import logger

try:
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None


class DenseIndex:
    """A simple dense index for cosine similarity search."""

    def __init__(self, dim: int, use_faiss: bool = True) -> None:
        self.dim = dim
        self.use_faiss = use_faiss and faiss is not None
        self._faiss_index = None
        self._mat: np.ndarray | None = None

    def build(self, embs: np.ndarray) -> None:
        """Build the index from normalized embeddings (shape: [N, dim])."""
        assert embs.ndim == 2 and embs.shape[1] == self.dim, "Embedding dimension mismatch."
        if self.use_faiss:
            logger.info("Building FAISS index, dim={}", self.dim)
            self._faiss_index = faiss.IndexFlatIP(self.dim)  # cosine with normalized vectors
            self._faiss_index.add(embs.astype(np.float32))
        else:
            logger.info("Using NumPy brute-force index.")
            self._mat = embs.astype(np.float32)

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-k for queries q (normalized)."""
        if self.use_faiss and self._faiss_index is not None:
            D, I = self._faiss_index.search(q.astype(np.float32), k)
            return D, I
        assert self._mat is not None, "Index not built."
        sims = q @ self._mat.T  # (Q, N)
        I = np.argpartition(-sims, kth=min(k, sims.shape[1]-1), axis=1)[:, :k]
        # reorder exact
        row = np.arange(q.shape[0])[:, None]
        top = sims[row, I]
        order = np.argsort(-top, axis=1)
        I = I[row, order]
        D = sims[row, I]
        return D, I
