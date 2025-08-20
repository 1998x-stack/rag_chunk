# -*- coding: utf-8 -*-
"""Sentence-Transformer embedder wrapper."""

from __future__ import annotations

from typing import List

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


class SentenceEmbedder:
    """Thin wrapper around Sentence-Transformer with sane defaults."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: {}", model_name)
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        embs = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        if normalize:
            # L2 归一化
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / norms
        return embs
