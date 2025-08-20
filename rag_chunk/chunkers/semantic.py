# -*- coding: utf-8 -*-
"""Semantic chunker using adjacent-sentence similarity drops (Method 4)."""

from __future__ import annotations

from typing import List

import numpy as np
import regex as re

from rag_chunk.retrieval.embedder import SentenceEmbedder
from rag_chunk.tokenization import count_tokens
from rag_chunk.types_ import Chunk, Document
from rag_chunk.chunkers.base import BaseChunker


_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;。])\s+|(?<=\.)\s+|\n{2,}", re.UNICODE)


class SemanticChunker(BaseChunker):
    """Detect topic shifts by cosine similarity dips between adjacent sentences."""

    name = "semantic"

    def __init__(self, embedder: SentenceEmbedder, max_tokens: int = 360,
                 sim_threshold: float = 0.60, window_size: int = 2) -> None:
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.sim_threshold = sim_threshold
        self.window_size = window_size

    def _sentences(self, text: str) -> List[str]:
        parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
        return parts or [text]

    def chunk(self, doc: Document) -> List[Chunk]:
        sents = self._sentences(doc.text)
        if len(sents) == 1:
            return [Chunk(doc_id=doc.doc_id, chunk_id=0, text=doc.text, start_char=0, end_char=len(doc.text))]

        # 句向量
        embs = self.embedder.encode(sents, batch_size=64, normalize=True)
        # 邻接相似度
        sims = (embs[:-1] * embs[1:]).sum(axis=1)
        # 平滑（窗口平均）
        if self.window_size > 1:
            kernel = np.ones(self.window_size) / self.window_size
            sims = np.convolve(sims, kernel, mode="same")
        # 以相似度低于阈值处作为潜在边界
        boundaries = {0}
        for i, s in enumerate(sims, start=1):
            if float(s) < self.sim_threshold:
                boundaries.add(i)

        # 组装 chunk：再按 token 上限打包，避免过长
        chunks: List[Chunk] = []
        cid = 0
        buf: List[str] = []
        start_char = 0
        cur_char = 0
        for i, sent in enumerate(sents):
            tok = count_tokens(sent)
            # 若边界命中或超限则冲刷
            should_flush = (i in boundaries and buf) or (buf and count_tokens("\n".join(buf)) + tok > self.max_tokens)
            if should_flush:
                text = "\n".join(buf).strip()
                chunks.append(Chunk(doc_id=doc.doc_id, chunk_id=cid, text=text,
                                    start_char=start_char, end_char=start_char + len(text)))
                cid += 1
                start_char = start_char + len(text)
                buf = []
            buf.append(sent)

        if buf:
            text = "\n".join(buf).strip()
            chunks.append(Chunk(doc_id=doc.doc_id, chunk_id=cid, text=text,
                                start_char=start_char, end_char=start_char + len(text)))
        return chunks