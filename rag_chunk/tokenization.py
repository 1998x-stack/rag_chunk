# -*- coding: utf-8 -*-
"""Lightweight tokenization utilities.

中文说明：
    - 为了避免重依赖，这里提供一个轻量 tokenizer：
      * 英文：按空白切分
      * 中文：单字视作 token
      * 混合文本：综合规则
    - 若需更精确，可替换为 HuggingFace tokenizer（保留相同接口）。
"""

from __future__ import annotations

import regex as re
from typing import List


def simple_tokens(text: str) -> List[str]:
    """Split text into lightweight tokens for budgeting lengths.

    Args:
        text: 输入文本

    Returns:
        tokens: 近似 token 列表
    """
    # 中文字符单独切分 + 英文按词切分
    # 保留数字与常见标点用于估算长度
    cn_chars = r"\p{Script=Han}"
    word = r"[A-Za-z0-9_]+"
    punct = r"[\p{P}\p{S}]"
    pattern = re.compile(f"({cn_chars}|{word}|{punct})", re.UNICODE)
    toks = [m.group(0) for m in pattern.finditer(text)]
    return toks


def count_tokens(text: str) -> int:
    """Approximate token count."""
    return len(simple_tokens(text))