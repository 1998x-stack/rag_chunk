# -*- coding: utf-8 -*-
"""PDF loader using PyPDF2.

中文说明：
    - 仅适用于“文本层可提取”的 PDF；若为扫描件需要先 OCR。
"""

from __future__ import annotations

import os
from typing import List

from loguru import logger
from PyPDF2 import PdfReader

from rag_chunk.preprocess.cleaners import clean_text
from rag_chunk.types_ import Document


def load_pdf_files(data_dir: str) -> List[Document]:
    """Load .pdf files under data_dir.

    Args:
        data_dir: 数据目录

    Returns:
        文档列表
    """
    docs: List[Document] = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, fn)
            try:
                reader = PdfReader(path)
                pages = []
                for p in reader.pages:
                    pages.append(p.extract_text() or "")
                text = clean_text("\n\n".join(pages))
                doc_id = os.path.relpath(path, data_dir)
                meta = {"path": path, "type": "pdf", "num_pages": len(pages)}
                docs.append(Document(doc_id=doc_id, text=text, meta=meta))
            except Exception as e:  # noqa: BLE001
                logger.exception("Failed to read pdf: {}", path)
    logger.info("Loaded {} pdf documents.", len(docs))
    return docs
