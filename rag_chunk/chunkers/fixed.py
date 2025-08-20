# -*- coding: utf-8 -*-
"""Fixed-size chunker with overlap (Method 1)."""

from __future__ import annotations

from typing import List

from rag_chunk.tokenization import simple_tokens, count_tokens
from rag_chunk.types_ import Chunk, Document
from rag_chunk.chunkers.base import BaseChunker


class FixedChunker(BaseChunker):
    """Fixed-size chunker.

    中文说明：
        - 按近似 token 等距切分，支持 overlap。
        - 通过字符索引回填 start/end，避免切断多字节字符。
    """

    name = "fixed"

    def __init__(self, max_tokens: int = 360, overlap: int = 60) -> None:
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, doc: Document) -> List[Chunk]:
        toks = simple_tokens(doc.text)
        chunks: List[Chunk] = []
        if not toks:
            return chunks

        step = max(1, self.max_tokens - self.overlap)
        i = 0
        cid = 0
        # 建立 token -> char 索引映射以回填 char span
        positions: List[int] = []
        cur = 0
        for t in toks:
            idx = doc.text.find(t, cur)
            idx = len(doc.text) if idx < 0 else idx
            positions.append(idx)
            cur = idx + len(t)

        while i < len(toks):
            j = min(i + self.max_tokens, len(toks))
            start_char = positions[i] if i < len(positions) else 0
            end_char = positions[j - 1] + len(toks[j - 1]) if j - 1 < len(positions) else len(doc.text)
            text = doc.text[start_char:end_char]
            chunks.append(Chunk(doc_id=doc.doc_id, chunk_id=cid, text=text, start_char=start_char, end_char=end_char))
            cid += 1
            i += step
        return chunks
