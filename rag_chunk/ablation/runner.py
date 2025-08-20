# -*- coding: utf-8 -*-
"""Ablation pipeline to compare chunking strategies end-to-end.

流程：
    1) 加载 data/ 下 txt/pdf 文档 -> Document
    2) 使用不同 chunker 切块 -> Chunk
    3) 构建倒排映射与 (doc_id, chunk_id) 顺序表
    4) 句子级采样生成“自监督查询”，并标记 gold (doc_id, chunk_id)
    5) 向量化 chunk 与 query，建立向量索引
    6) 检索评估：Success@k、MRR、nDCG@k
    7) （可选）应用方法11：查询时动态扩窗，观察指标变化
    8) 输出 CSV 报告

注意：
    - 所有日志使用 loguru，并保存在 work_dir/logs。
"""

from __future__ import annotations

import csv
import os
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import regex as re
from loguru import logger
from tqdm import tqdm

from rag_chunk.config import AppConfig
from rag_chunk.types_ import Chunk, Document, Query, RetrievalItem
from rag_chunk.chunkers import BaseChunker, FixedChunker, RecursiveSepChunker, SemanticChunker, SentencePackChunker
from rag_chunk.retrieval.embedder import SentenceEmbedder
from rag_chunk.retrieval.index_store import DenseIndex
from rag_chunk.retrieval.query_expand import expand_neighbors


_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;。])\s+|(?<=\.)\s+|\n{2,}", re.UNICODE)


def sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or [text]


def build_chunker(name: str, cfg: AppConfig, embedder: SentenceEmbedder | None) -> BaseChunker:
    name = name.lower()
    if name == "fixed":
        return FixedChunker(max_tokens=cfg.chunker.max_tokens, overlap=cfg.chunker.overlap)
    if name == "sentence":
        return SentencePackChunker(max_tokens=cfg.chunker.max_tokens, min_sent_len=cfg.chunker.min_sent_len)
    if name == "recursive":
        return RecursiveSepChunker(max_tokens=cfg.chunker.max_tokens, seps=cfg.chunker.sep_headers)
    if name == "semantic":
        assert embedder is not None, "Semantic chunker requires an embedder."
        return SemanticChunker(
            embedder=embedder,
            max_tokens=cfg.chunker.max_tokens,
            sim_threshold=cfg.chunker.sim_threshold,
            window_size=cfg.chunker.window_size,
        )
    raise ValueError(f"Unknown chunker: {name}")


def make_queries(docs: List[Document], per_doc: int) -> List[Tuple[str, str]]:
    """Sample (doc_id, sentence) pairs as queries."""
    rng = random.Random(42)
    pairs: List[Tuple[str, str]] = []
    for d in docs:
        sents = sentences(d.text)
        if not sents:
            continue
        samples = sents if len(sents) <= per_doc else rng.sample(sents, per_doc)
        for s in samples:
            pairs.append((d.doc_id, s))
    return pairs


def locate_gold_chunk(chunks: List[Chunk], sentence: str) -> int:
    """Find which chunk contains the sentence (first match)."""
    for c in chunks:
        if sentence and c.text.find(sentence) >= 0:
            return c.chunk_id
    # 未命中则返回最相似？这里返回 -1 表示不可用样本
    return -1


def success_at_k(scores: List[List[RetrievalItem]], golds: List[Tuple[str, int]], k: int) -> float:
    ok = 0
    total = 0
    for items, (gdoc, gcid) in zip(scores, golds):
        total += 1
        for h in items[:k]:
            if h.doc_id == gdoc and h.chunk_id == gcid:
                ok += 1
                break
    return ok / max(1, total)


def mrr(scores: List[List[RetrievalItem]], golds: List[Tuple[str, int]]) -> float:
    rr = []
    for items, (gdoc, gcid) in zip(scores, golds):
        rank = None
        for i, h in enumerate(items, start=1):
            if h.doc_id == gdoc and h.chunk_id == gcid:
                rank = i
                break
        rr.append(1.0 / rank if rank else 0.0)
    return float(np.mean(rr)) if rr else 0.0


def ndcg_at_k(scores: List[List[RetrievalItem]], golds: List[Tuple[str, int]], k: int) -> float:
    def dcg(rels: List[int]) -> float:
        return sum((rel / np.log2(i + 2)) for i, rel in enumerate(rels))

    vals = []
    for items, (gdoc, gcid) in zip(scores, golds):
        rels = [1 if (h.doc_id == gdoc and h.chunk_id == gcid) else 0 for h in items[:k]]
        ideal = sorted(rels, reverse=True)
        vals.append(dcg(rels) / (dcg(ideal) + 1e-12))
    return float(np.mean(vals)) if vals else 0.0


def run_ablation(
    work_dir: str,
    data_dir: str,
    chunker_names: List[str],
    cfg: AppConfig,
    docs: List[Document],
) -> None:
    """Main ablation loop for given chunkers."""

    os.makedirs(os.path.join(work_dir, "reports"), exist_ok=True)

    # Embedding model (shared)
    embedder = SentenceEmbedder(cfg.retrieval.model_name)

    # Precompute queries
    raw_pairs = make_queries(docs, per_doc=cfg.eval.num_queries_per_doc)
    logger.info("Prepared {} raw queries.", len(raw_pairs))

    # For each chunker:
    for cname in chunker_names:
        logger.info("==== Chunker: {} ====", cname)
        chunker = build_chunker(cname, cfg, embedder if cname == "semantic" else None)

        # Chunk all documents
        doc2chunks: Dict[str, List[Chunk]] = {}
        all_chunks: List[Chunk] = []
        for d in tqdm(docs, desc=f"Chunking[{cname}]"):
            chs = chunker.chunk(d)
            doc2chunks[d.doc_id] = chs
            all_chunks.extend(chs)
        logger.info("Total chunks: {} | avg per doc: {:.2f}", len(all_chunks), len(all_chunks)/max(1, len(docs)))

        # Build mapping for retrieval results back
        chunk_keys = [(c.doc_id, c.chunk_id) for c in all_chunks]

        # Build chunk embeddings
        chunk_texts = [c.text for c in all_chunks]
        chunk_embs = embedder.encode(chunk_texts, batch_size=cfg.retrieval.batch_size, normalize=True)
        index = DenseIndex(dim=chunk_embs.shape[1], use_faiss=cfg.retrieval.use_faiss)
        index.build(chunk_embs)

        # Build queries with gold chunk ids
        queries: List[Query] = []
        q_texts: List[str] = []
        for i, (doc_id, qtxt) in enumerate(raw_pairs):
            chs = doc2chunks.get(doc_id, [])
            gcid = locate_gold_chunk(chs, qtxt)
            if gcid >= 0:
                queries.append(Query(qid=f"{cname}-{i}", query_text=qtxt, gold_doc_id=doc_id, gold_chunk_id=gcid))
                q_texts.append(qtxt)
        logger.info("Valid queries for {}: {}", cname, len(queries))

        if not queries:
            logger.warning("No valid queries for {}. Skipping.", cname)
            continue

        # Encode queries & search
        q_embs = embedder.encode(q_texts, batch_size=cfg.retrieval.batch_size, normalize=True)
        D, I = index.search(q_embs, k=max(cfg.retrieval.k, 50))  # 先取更多，后面可扩窗和评估
        # Convert to hits
        hits: List[List[RetrievalItem]] = []
        for row in I:
            items = []
            for idx in row:
                doc_id, cid = chunk_keys[idx]
                items.append(RetrievalItem(doc_id=doc_id, chunk_id=cid, score=1.0))
            hits.append(items)

        # Optional: apply method 11 (expand neighbors)
        if cfg.expand.enable:
            expanded: List[List[RetrievalItem]] = []
            for items in hits:
                expanded.append(expand_neighbors(items[:cfg.retrieval.k], doc2chunks, neighbors=cfg.expand.neighbors))
            hits_eval = expanded
        else:
            hits_eval = [items[:cfg.retrieval.k] for items in hits]

        golds = [(q.gold_doc_id, q.gold_chunk_id) for q in queries]
        s_at_k = success_at_k(hits_eval, golds, k=cfg.retrieval.k)
        _mrr = mrr(hits_eval, golds)
        _ndcg = ndcg_at_k(hits_eval, golds, k=min(cfg.eval.ndcg_k, cfg.retrieval.k))

        logger.info("Metrics[{}]: Success@{}={:.4f} | MRR={:.4f} | nDCG@{}={:.4f}",
                    cname, cfg.retrieval.k, s_at_k, _mrr, cfg.eval.ndcg_k, _ndcg)

        # Write report
        report_path = os.path.join(work_dir, "reports", f"{cname}_report.csv")
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["chunker", "k", "neighbors", "success@k", "mrr", "ndcg@k", "num_docs", "num_chunks", "num_queries"])
            w.writerow([cname, cfg.retrieval.k, cfg.expand.neighbors if cfg.expand.enable else 0,
                        f"{s_at_k:.4f}", f"{_mrr:.4f}", f"{_ndcg:.4f}",
                        len(docs), len(all_chunks), len(queries)])
        logger.info("Report written: {}", report_path)
