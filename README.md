# RAG Paper QA Data Package

This repository contains the current data-engineering deliverables for a literature-grounded RAG project built with LlamaIndex.

The package includes:

- a cleaned page-level corpus built from 10 arXiv papers
- two chunked datasets (`256` and `512` token settings)
- two persisted vector indexes for the main corpus
- manifests, quality-check outputs, and an evaluation seed set
- reproducible scripts for chunking, indexing, and quality checking

## Project Context

The larger course project is a practical LLM application built with LlamaIndex. The current direction is a literature-grounded RAG assistant that can answer questions about RAG-related research papers, compare methods across papers, and return grounded evidence instead of unsupported free-form responses.

This repository is the data-engineering baseline for that system. It contains the prepared corpus, metadata, chunk variants, indexes, and evaluation seeds that downstream teammates can directly build on.

## Scope Decision

The current handoff uses only the main research-paper corpus:

- `10` arXiv papers on RAG, document-grounded QA, retrieval evaluation, and practical RAG system design

Therefore:

- all current chunks are generated from the paper corpus only
- all current indexes are built from the paper corpus only
- all current evaluation seeds target the paper corpus only

## Repository Layout

```text
config.yaml
docs/
  interface_guide.md
scripts/
  chunk.py
  build_index.py
  embedding_utils.py
  quality_check.py
data/
  raw/main/
  cleaned/cleaned_pages.jsonl
  chunks/chunks_256.jsonl
  chunks/chunks_512.jsonl
  indexes/index_main_256/
  indexes/index_main_512/
  manifests/source_manifest.json
  manifests/index_manifest.json
  quality/data_quality_report.md
  eval/eval_seed_set.jsonl
```

## Corpus Summary

- Documents: `10`
- Cleaned page records: `143`
- Indexable pages: `109`
- Indexable tokens: `86,297`
- Chunks (`256`): `379`
- Chunks (`512`): `192`

## What Has Been Completed

The data-engineering work completed in this package includes:

- source selection and manifest creation for the main paper corpus
- page-level cleaning with metadata such as `page_type`, `section_path`, and `excluded_from_index`
- exact token-based chunking with two settings: `256/26` and `512/51`
- vector index construction for `index_main_256` and `index_main_512`
- quality checks for chunk integrity and metadata completeness
- creation of an initial evidence-linked evaluation seed set

These artifacts are intended to let downstream teammates continue the project without rebuilding the data layer from scratch.

## Quick Start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional local-package mode:

```bash
pip install --target .deps -r requirements.txt
```

## Main Entry Points

Generate chunks:

```bash
python3 scripts/chunk.py --config config.yaml
```

Build indexes:

```bash
python3 scripts/build_index.py --config config.yaml --corpus main
```

Run quality checks:

```bash
python3 scripts/quality_check.py --config config.yaml
```

## Index Loading

The persisted indexes use a custom embedding wrapper instead of the default LlamaIndex HuggingFace embedding class. Downstream code should import:

```python
from scripts.embedding_utils import TransformersBgeEmbedding
```

For full loading and retrieval examples, see:

- `docs/interface_guide.md`

## Notes for Downstream Teammates

- Use only `corpus = "main"` for experiments and evaluation.
- The chunk files already exclude pages marked `excluded_from_index = true`.
- `index_manifest.json` records the exact chunk setting, embedding model, and runtime LlamaIndex version used to build each index.
- `eval_seed_set.jsonl` provides an initial evaluation set with evidence pointers.

## Recommended Next Steps

- Modeling teammates should start from `data/indexes/index_main_256` and `data/indexes/index_main_512`, following `docs/interface_guide.md`.
- Evaluation teammates should use `data/chunks/`, `data/quality/data_quality_report.md`, and `data/eval/eval_seed_set.jsonl`.
- Integration teammates can build the UI and retrieval demo on top of the main-paper indexes first.

## Deliverable Status

This package is intended as the current handoff-ready baseline for the data-engineering stage.

## 5. Application Integration & Quick Start (Full RAG System)

This section describes how to integrate the data package with the full RAG QA system, including model deployment, index loading, and web UI startup.

## 5.1 Project Structure (Full System)

```
rag-paper-qa/
├── app.py                  # Web UI entry (Gradio)
├── manager.py              # Model deployment & LLM manager
├── engine.py               # RAG index query engine
├── config.py               # Global model & path config
├── rebuild_index.py        # Index rebuild script (fix dimension mismatch)
├── requirements.txt         # Full dependencies
│
├── data/                   # Provided data package
├── scripts/                # Provided data utilities
└── docs/
```

## 5.2 Core Configuration Files

### `config.py` (Global System Config)

```
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LLM Models
MISTRAL_MODEL = "mistral:7b-instruct-v0.2-q4_K_M"
T5_MODEL = "google/flan-t5-base"

# Index Paths
INDEX_PATH_256 = "data/indexes/index_main_256"
INDEX_PATH_512 = "data/indexes/index_main_512"
DEFAULT_INDEX_DIR = INDEX_PATH_256
SIMILARITY_TOP_K = 3
```

### `manager.py` (Model Manager)

Manages embedding model loading and LLM backend switching:

- Embedding: `TransformersBgeEmbedding` (matches data package)
- LLMs: Mistral (Ollama) / Flan-T5 (local pipeline)
- Dimension alignment: 768-dim for index compatibility

### `engine.py` (RAG Query Engine)

Loads pre-built indexes and supports:

- Similarity retrieval
- Question answering
- Source evidence return

## Full Environment Setup

Install required packages (using Tsinghua mirror for stability):

```
pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install llama-index-llms-ollama gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Install Ollama (for Mistral 7B)

```
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b-instruct-v0.2-q4_K_M
# Windows download .exe file from https://ollama.com/download
ollama pull mistral:7b-instruct-v0.2-q4_K_M
```

## 5.4 Critical Step: Rebuild Index (Fix Dimension Mismatch)

The original indexes use fixed embedding dimensions. Run this **once** to rebuild compatible indexes:

```
python rebuild_index.py
```

This ensures embedding dimension alignment (768 ↔ 768) and eliminates:

```
ValueError: shapes (384,) and (768,) not aligned
```

## 5.5 Launch Web UI

```
python app.py
```

Open your browser:

- http://127.0.0.1:7865

## 5.6 Web UI Features

- Dual LLM support: `mistral` / `t5`
- Dual chunk sizes: `256` / `512`
- Real-time answer generation
- Full source evidence display
- Offline keyword retrieval fallback

## 5.7 Example Questions

- What is the role of the retriever in RAG?
- How does CRAG correct hallucinations?
- What is the difference between DPR and BM25?
- What are the key components of a practical RAG system?

## 5.8 Troubleshooting

### 1. ModuleNotFoundError: `llama_index.llms.ollama`

```
pip install llama-index-llms-ollama
```

### 2. Dimension mismatch: `shapes (384,)` & `(768,)`

Run index rebuild:

```
python rebuild_index.py
```

### 3. Hugging Face connection timeout

Use offline local retrieval mode or check network.

### 4. Index failed to load

Ensure `data/indexes/` exists and run `rebuild_index.py`.

## 5.9 Full System Workflow

1. Load embedding model (from `scripts/embedding_utils.py`)
2. Load rebuilt vector index
3. Accept user question via web UI
4. Retrieve relevant chunks
5. Generate answer using selected LLM
6. Display answer + source evidence

## 5.10 Notes for Developers

- Do NOT modify `scripts/embedding_utils.py`
- Use `index_main_256` for speed, `index_main_512` for context
- Fallback to local JSONL retrieval if models are unavailable
- All data remains strictly within the 10 arXiv paper corpus

------

## 6. Final System Summary

This full RAG QA system integrates:

- Cleaned & chunked paper corpus
- Persisted vector indexes
- Custom embedding class
- Dual LLM backends
- Interactive web UI
- Full source attribution

The system is ready for deployment, evaluation, and extension.
