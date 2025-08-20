# -*- coding: utf-8 -*-
"""Plain text loader."""

from __future__ import annotations

import io
import os
from typing import List

from loguru import logger

from rag_chunk.preprocess.cleaners import clean_text
from rag_chunk.types_ import Document


def load_txt_files(data_dir: str) -> List[Document]:
    """Load .txt files under data_dir.

    Args:
        data_dir: 数据目录

    Returns:
        文档列表
    """
    docs: List[Document] = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.lower().endswith(".txt"):
                continue
            path = os.path.join(root, fn)
            try:
                with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                text = clean_text(raw)
                doc_id = os.path.relpath(path, data_dir)
                docs.append(Document(doc_id=doc_id, text=text, meta={"path": path, "type": "txt"}))
            except Exception as e:  # noqa: BLE001
                logger.exception("Failed to read txt: {}", path)
    logger.info("Loaded {} txt documents.", len(docs))
    return docs
