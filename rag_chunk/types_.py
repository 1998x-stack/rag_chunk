# -*- coding: utf-8 -*-
"""Common datatypes for documents and chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Document:
    """A raw document loaded from disk.

    Attributes:
        doc_id: 文档唯一标识（文件相对路径）。
        text: 文本内容。
        meta: 其他元数据（页码映射、来源等）。
    """

    doc_id: str
    text: str
    meta: Optional[dict] = None


@dataclass
class Chunk:
    """A contiguous text span associated with a document.

    Attributes:
        doc_id: 来源文档 ID。
        chunk_id: 文档内块编号（顺序号，从 0 开始）。
        text: 块文本。
        start_char: 在原文中的起始字符索引。
        end_char: 在原文中的结束字符索引（闭区间外的 end）。
    """

    doc_id: str
    chunk_id: int
    text: str
    start_char: int
    end_char: int


@dataclass
class RetrievalItem:
    """A single retrieval result."""

    doc_id: str
    chunk_id: int
    score: float


@dataclass
class Query:
    """A query record used in evaluation."""

    qid: str
    query_text: str
    gold_doc_id: str
    gold_chunk_id: int