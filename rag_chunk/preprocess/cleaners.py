# -*- coding: utf-8 -*-
"""Text cleaners and normalization steps.

中文说明：
    - 统一做空白归一化、连字符回填、常见页眉/页脚/空行清理。
"""

from __future__ import annotations

import regex as re


def normalize_spaces(text: str) -> str:
    """Normalize whitespace and line breaks.

    Args:
        text: 原始文本

    Returns:
        归一化后的文本
    """
    # 替换 Windows 换行
    text = text.replace("\r\n", "\n")
    # 多空白压缩
    text = re.sub(r"[ \t]+", " ", text)
    # 连续多空行折叠为双换行（保留段落感）
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dehyphenate(text: str) -> str:
    """Fix hyphenated line breaks like 'inter-\nnational' -> 'international'."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def clean_text(text: str) -> str:
    """Apply a sequence of cleaning steps."""
    text = dehyphenate(text)
    text = normalize_spaces(text)
    return text
