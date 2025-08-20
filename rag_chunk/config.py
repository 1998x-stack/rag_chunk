# -*- coding: utf-8 -*-
"""Global configuration and default hyperparameters.

This module centralizes default settings for chunkers, retriever, and evaluation.

中文说明：
    - 统一放置默认超参，方便命令行覆盖或在代码里按需修改。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ChunkerConfig:
    """Chunker-specific configuration.

    Attributes:
        max_tokens: 每块允许的最大 token 数。
        overlap: 邻接块之间的重叠 token 数（固定长度、部分策略可用）。
        min_sent_len: 句打包时的最小句长，过短可合并。
        sep_headers: 递归切块时的分隔符列表（如 Markdown 头部标记）。
        sim_threshold: 语义切块的相邻句相似度阈值（低于视为主题切换）。
        window_size: 语义切块平滑窗口，避免抖动。
    """

    max_tokens: int = 360
    overlap: int = 60
    min_sent_len: int = 10
    sep_headers: List[str] = field(default_factory=lambda: ["\n# ", "\n## ", "\n### ", "\n- ", "\n* "])
    sim_threshold: float = 0.60
    window_size: int = 2


@dataclass
class RetrievalConfig:
    """Retriever & indexing configuration.

    Attributes:
        k: 检索的 top-k。
        use_faiss: 是否优先使用 FAISS；否则回退为 NumPy 暴力检索。
        model_name: Sentence-Transformer 模型名（多语言）。
        batch_size: 向量化批大小。
    """

    k: int = 8
    use_faiss: bool = True
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 64


@dataclass
class ExpandConfig:
    """Query-time expansion configuration (Method 11).

    Attributes:
        neighbors: 命中块左右扩展的相邻块数量。
        enable: 是否启用动态扩窗。
    """

    neighbors: int = 1
    enable: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        num_queries_per_doc: 每份文档采样多少条“自监督查询”（从句子采样）。
        ndcg_k: nDCG 的 k。
    """

    num_queries_per_doc: int = 30
    ndcg_k: int = 8


@dataclass
class AppConfig:
    """Top-level config container."""

    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    expand: ExpandConfig = field(default_factory=ExpandConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)