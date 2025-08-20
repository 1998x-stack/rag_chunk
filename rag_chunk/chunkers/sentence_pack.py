# -*- coding: utf-8 -*-
"""Sentence/Paragraph packing chunker (Method 2)."""

from __future__ import annotations

from typing import List

import regex as re

from rag_chunk.tokenization import count_tokens
from rag_chunk.types_ import Chunk, Document
from rag_chunk.chunkers.base import BaseChunker


_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;。])\s+|(?<=\.)\s+|\n{2,}", re.UNICODE)


def split_sentences(text: str) -> List[str]:
    """Rudimentary sentence/paragraph splitter for mixed Chinese/English."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts


class SentencePackChunker(BaseChunker):
    """Pack sentences/paragraphs into token buckets."""

    name = "sentence"

    def __init__(self, max_tokens: int = 360, min_sent_len: int = 10) -> None:
        self.max_tokens = max_tokens
        self.min_sent_len = min_sent_len

    def chunk(self, doc: Document) -> List[Chunk]:
        sents = split_sentences(doc.text) or [doc.text]
        chunks: List[Chunk] = []
        buf: List[str] = []
        buf_tok = 0
        cid = 0
        char_cursor = 0

        def flush(buffer: List[str], start_char: int, cid: int) -> Chunk:
            text = "\n".join(buffer).strip()
            end_char = start_char + len(text)
            return Chunk(doc_id=doc.doc_id, chunk_id=cid, text=text, start_char=start_char, end_char=end_char)

        for s in sents:
            stoks = max(count_tokens(s), self.min_sent_len)
            # 若加入后超过上限，先冲刷
            if buf and buf_tok + stoks > self.max_tokens:
                chunk = flush(buf, char_cursor, cid)
                chunks.append(chunk)
                cid += 1
                char_cursor = chunk.end_char
                buf, buf_tok = [], 0
            buf.append(s)
            buf_tok += stoks

        if buf:
            chunks.append(Chunk(doc_id=doc.doc_id, chunk_id=cid, text="\n".join(buf).strip(),
                                start_char=char_cursor, end_char=char_cursor + len("\n".join(buf).strip())))
        return chunks
