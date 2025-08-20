# -*- coding: utf-8 -*-
"""Command line entry for RAG chunk ablation study."""

from __future__ import annotations

import argparse
import os
from typing import List

from loguru import logger

from rag_chunk.config import AppConfig, ChunkerConfig, EvalConfig, ExpandConfig, RetrievalConfig
from rag_chunk.logging_utils import init_logger
from rag_chunk.loaders.pdf_loader import load_pdf_files
from rag_chunk.loaders.txt_loader import load_txt_files
from rag_chunk.ablation.runner import run_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Chunk Ablation CLI")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory (txt/pdf).")
    parser.add_argument("--work_dir", type=str, required=True, help="Path to work/output directory.")
    parser.add_argument("--chunkers", nargs="+", default=["fixed", "sentence", "recursive", "semantic"],
                        help="Chunker names to run. Options: fixed sentence recursive semantic")
    parser.add_argument("--k", type=int, default=8, help="Top-k for retrieval.")
    parser.add_argument("--neighbors", type=int, default=1, help="Neighbors for query-time expansion (Method 11).")
    parser.add_argument("--do_expand", type=str, default="true", help="Enable query-time expansion.")
    parser.add_argument("--max_tokens", type=int, default=360, help="Max tokens per chunk.")
    parser.add_argument("--overlap", type=int, default=60, help="Overlap for fixed chunker.")
    parser.add_argument("--min_sent_len", type=int, default=10, help="Minimum sentence length when packing.")
    parser.add_argument("--sim_threshold", type=float, default=0.60, help="Similarity threshold for semantic chunking.")
    parser.add_argument("--window_size", type=int, default=2, help="Smoothing window for semantic chunking.")
    parser.add_argument("--model_name", type=str,
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="Sentence-Transformer model name.")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--num_queries_per_doc", type=int, default=30, help="Self-supervised samples per doc.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    init_logger(args.work_dir)

    # Compose config
    cfg = AppConfig(
        chunker=ChunkerConfig(
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            min_sent_len=args.min_sent_len,
            sim_threshold=args.sim_threshold,
            window_size=args.window_size,
        ),
        retrieval=RetrievalConfig(
            k=args.k,
            use_faiss=True,
            model_name=args.model_name,
            batch_size=args.batch_size,
        ),
        expand=ExpandConfig(
            neighbors=args.neighbors,
            enable=str(args.do_expand).lower() in {"true", "1", "yes", "y"},
        ),
        eval=EvalConfig(
            num_queries_per_doc=args.num_queries_per_doc,
            ndcg_k=args.k,
        ),
    )

    # Load docs
    docs_txt = load_txt_files(args.data_dir)
    docs_pdf = load_pdf_files(args.data_dir)
    docs = docs_txt + docs_pdf
    if not docs:
        logger.error("No documents found under {}. Please put .txt/.pdf into it.", args.data_dir)
        return

    # Run ablation
    run_ablation(
        work_dir=args.work_dir,
        data_dir=args.data_dir,
        chunker_names=args.chunkers,
        cfg=cfg,
        docs=docs,
    )


if __name__ == "__main__":
    main()
