# -*- coding: utf-8 -*-
"""Recursive/separator-based chunker (Method 3)."""

from __future__ import annotations

from typing import List

import regex as re

from rag_chunk.tokenization import count_tokens
from rag_chunk.types_ import Chunk, Document
from rag_chunk.chunkers.base import BaseChunker


class RecursiveSepChunker(BaseChunker):
    """Respect headings/lists first, then pack to token budget."""

    name = "recursive"

    def __init__(self, max_tokens: int = 360, seps: List[str] | None = None) -> None:
        self.max_tokens = max_tokens
        self.seps = seps or ["\n# ", "\n## ", "\n### ", "\n- ", "\n* ", "\n1. "]

    def _split_by_seps(self, text: str) -> List[str]:
        # 优先按 Markdown 样式切；若不命中，按双换行退化
        pat = "|".join(re.escape(s) for s in self.seps)
        blocks: List[str] = []
        if re.search(pat, text):
            # 在分隔符处切，但保留分隔符行
            parts = re.split(f"({pat})", text)
            cur = ""
            for i in range(0, len(parts), 2):
                sep = parts[i + 1] if i + 1 < len(parts) else ""
                seg = parts[i]
                if sep:
                    if cur.strip():
                        blocks.append(cur)
                    cur = (sep + seg)
                else:
                    cur += seg
            if cur.strip():
                blocks.append(cur)
        else:
            blocks = [b for b in text.split("\n\n") if b.strip()]
        return blocks

    def chunk(self, doc: Document) -> List[Chunk]:
        blocks = self._split_by_seps(doc.text)
        chunks: List[Chunk] = []
        cid = 0
        char_cursor = 0
        buf: List[str] = []
        buf_tok = 0

        def flush(buffer: List[str], start_char: int, cid: int) -> Chunk:
            text = "\n\n".join(buffer).strip()
            end_char = start_char + len(text)
            return Chunk(doc_id=doc.doc_id, chunk_id=cid, text=text, start_char=start_char, end_char=end_char)

        for b in blocks:
            btok = count_tokens(b)
            if btok >= self.max_tokens:
                # 超长块继续按段落细分
                paras = [p for p in b.split("\n") if p.strip()]
                for p in paras:
                    ptok = count_tokens(p)
                    if buf and buf_tok + ptok > self.max_tokens:
                        chunk = flush(buf, char_cursor, cid)
                        chunks.append(chunk)
                        cid += 1
                        char_cursor = chunk.end_char
                        buf, buf_tok = [], 0
                    buf.append(p)
                    buf_tok += ptok
            else:
                if buf and buf_tok + btok > self.max_tokens:
                    chunk = flush(buf, char_cursor, cid)
                    chunks.append(chunk)
                    cid += 1
                    char_cursor = chunk.end_char
                    buf, buf_tok = [], 0
                buf.append(b)
                buf_tok += btok

        if buf:
            chunks.append(Chunk(doc_id=doc.doc_id, chunk_id=cid, text="\n\n".join(buf).strip(),
                                start_char=char_cursor, end_char=char_cursor + len("\n\n".join(buf).strip())))
        return chunks
