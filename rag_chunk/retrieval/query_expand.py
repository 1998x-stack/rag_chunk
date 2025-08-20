# -*- coding: utf-8 -*-
"""Query-time dynamic expansion (Method 11).

中文说明：
    - 在检索命中后，按 (neighbors) 扩展左右相邻块，拼接为更完整上下文。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from rag_chunk.types_ import Chunk, RetrievalItem


def expand_neighbors(
    hits: List[RetrievalItem],
    doc2chunks: Dict[str, List[Chunk]],
    neighbors: int = 1,
) -> List[RetrievalItem]:
    """Expand retrieval items by adding left/right neighbors as new pseudo-hits.

    Note:
        - 扩展后的“伪命中”共享同一分数（或可按衰减因子减权）。
        - 保证 (doc_id, chunk_id) 去重。

    Args:
        hits: 原始命中列表
        doc2chunks: 文档到块序列的映射
        neighbors: 扩展邻居数

    Returns:
        扩展后的命中列表
    """
    seen = {(h.doc_id, h.chunk_id) for h in hits}
    out = list(hits)
    for h in hits:
        chunks = doc2chunks.get(h.doc_id, [])
        for offset in range(1, neighbors + 1):
            for nid in (h.chunk_id - offset, h.chunk_id + offset):
                if 0 <= nid < len(chunks):
                    key = (h.doc_id, nid)
                    if key not in seen:
                        seen.add(key)
                        out.append(RetrievalItem(doc_id=h.doc_id, chunk_id=nid, score=h.score))
    return out