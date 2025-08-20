# rag_chunk

## 项目结构（File Category）
```
rag_chunk_lab/
├─ README.md
├─ requirements.txt
├─ pyproject.toml                 # 可选：严格格式/构建
├─ rag_chunk/
│  ├─ __init__.py
│  ├─ config.py                   # 全局配置/默认超参
│  ├─ logging_utils.py            # loguru 封装
│  ├─ types_.py                   # 通用数据结构定义
│  ├─ tokenization.py             # 轻量 tokenizer（兼容中英文）
│  ├─ preprocess/
│  │  ├─ __init__.py
│  │  └─ cleaners.py              # 文本清洗与段落归一化
│  ├─ loaders/
│  │  ├─ __init__.py
│  │  ├─ txt_loader.py
│  │  └─ pdf_loader.py
│  ├─ chunkers/
│  │  ├─ __init__.py
│  │  ├─ base.py                  # 抽象基类
│  │  ├─ fixed.py                 # 方法1：固定长度+重叠
│  │  ├─ sentence_pack.py         # 方法2：句/段落打包
│  │  ├─ recursive_sep.py         # 方法3：递归/分隔符
│  │  └─ semantic.py              # 方法4：语义切块（相邻句相似度突变）
│  ├─ retrieval/
│  │  ├─ __init__.py
│  │  ├─ embedder.py              # Sentence-Transformer 封装
│  │  ├─ index_store.py           # FAISS/NumPy 索引封装
│  │  └─ query_expand.py          # 方法11：查询时动态扩窗
│  ├─ ablation/
│  │  ├─ __init__.py
│  │  └─ runner.py                # 端到端构建/检索/评估/A/B
│  └─ cli.py                      # 命令行入口：一键跑 ablation
└─ data/
   └─ ...                         # 你的 txt/pdf 数据
```

An industrial-grade, pluggable RAG chunking and ablation toolkit covering:
- Method 1: Fixed-size with overlap
- Method 2: Sentence/Paragraph packing
- Method 3: Recursive separator-based (Markdown/blank lines)
- Method 4: Semantic chunking (adjacent-sentence similarity break)
- Method 11: Query-time dynamic window expansion

## Quick Start

```bash
# 1) Create venv and install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Put .txt/.pdf under data/
tree data/

# 3) Run ablation with default configs
python -m rag_chunk.cli --data_dir data --work_dir out \
  --chunkers fixed sentence recursive semantic \
  --do_expand true --k 8 --neighbors 1
```