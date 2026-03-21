# Data Interface Guide

This document explains how downstream teammates can consume the current data-engineering deliverables without needing additional clarification from the data team.

## 1. Project Scope and Handoff Context

This package supports a literature-grounded RAG assistant built with LlamaIndex. The current dataset is designed for querying and comparing RAG-related research papers rather than for general course-document QA.

Important scope decision:

- the active corpus in this handoff is the paper corpus only
- all current chunks, indexes, quality checks, and evaluation seeds are based only on the main paper corpus

What the data team has completed:

- collected and standardized the main paper corpus
- produced page-level cleaned records with structured metadata
- generated two token-based chunk variants
- built two persisted vector indexes for the main corpus
- ran quality checks on chunk integrity and metadata completeness
- prepared an initial evaluation seed set with evidence pointers

Downstream teammates should treat this package as the baseline data layer for the current project stage.

## 2. Directory Structure

Current project data layout:

```text
data/
├── raw/
│   └── main/
│       ├── beyond_relevance_rag_coverage_2026.pdf
│       ├── corrective_retrieval_augmented_generation_2024.pdf
│       ├── engineering_rag_systems_2025.pdf
│       ├── policy_document_rag_2026.pdf
│       ├── raft_domain_specific_rag_2024.pdf
│       ├── ragalyst_2025.pdf
│       ├── ragchecker_2024.pdf
│       ├── reason_and_verify_rag_2026.pdf
│       ├── retrievalqa_2024.pdf
│       └── topochunker_2026.pdf
├── cleaned/
│   └── cleaned_pages.jsonl
├── chunks/
│   ├── chunks_256.jsonl
│   └── chunks_512.jsonl
├── indexes/
│   ├── index_main_256/
│   │   ├── default__vector_store.json
│   │   ├── docstore.json
│   │   ├── graph_store.json
│   │   ├── image__vector_store.json
│   │   └── index_store.json
│   └── index_main_512/
│       ├── default__vector_store.json
│       ├── docstore.json
│       ├── graph_store.json
│       ├── image__vector_store.json
│       └── index_store.json
├── manifests/
│   ├── index_manifest.json
│   └── source_manifest.json
├── eval/
│   └── eval_seed_set.jsonl
└── quality/
    └── data_quality_report.md
```

Artifact purposes:

- `data/raw/main/*.pdf`: canonical raw PDF corpus for the main experimental dataset.
- `data/cleaned/cleaned_pages.jsonl`: page-level cleaned records with metadata.
- `data/chunks/chunks_256.jsonl`: chunk-level dataset built with 256-token windows and 26-token overlap.
- `data/chunks/chunks_512.jsonl`: chunk-level dataset built with 512-token windows and 51-token overlap.
- `data/indexes/index_main_256/`: persisted LlamaIndex SimpleVectorStore for the 256-token main corpus.
- `data/indexes/index_main_512/`: persisted LlamaIndex SimpleVectorStore for the 512-token main corpus.
- `data/manifests/source_manifest.json`: source PDF manifest.
- `data/manifests/index_manifest.json`: index build manifest used by downstream loading and experiment reporting.
- `data/eval/eval_seed_set.jsonl`: evaluation seed questions with evidence pointers for the evaluation teammate.
- `data/quality/data_quality_report.md`: chunk-level quality summary.

Related code locations:

- `scripts/chunk.py`: chunk generation.
- `scripts/build_index.py`: index construction.
- `scripts/embedding_utils.py`: custom embedding wrapper required for loading and querying indexes.
- `scripts/quality_check.py`: chunk quality checks and report generation.

## 3. `config.yaml` Field Guide

```yaml
tokenizer_name: cl100k_base
embedding_model: BAAI/bge-base-en-v1.5
chunk_configs:
  - chunk_size: 256
    overlap: 26
  - chunk_size: 512
    overlap: 51
paths:
  cleaned_pages: data/cleaned/cleaned_pages.jsonl
  chunks_dir: data/chunks
  indexes_dir: data/indexes
  manifests_dir: data/manifests
  quality_dir: data/quality
llamaindex_version: 0.14.18
vector_store_type: SimpleVectorStore
persist_backend: local_filesystem
```

Field meanings:

- `tokenizer_name`: tokenizer used for all token-count computations and chunk boundaries.
- `embedding_model`: embedding model name used when building or loading indexes.
- `chunk_configs`: available chunking variants.
- `paths.cleaned_pages`: page-level cleaned source file.
- `paths.chunks_dir`: location of chunked JSONL outputs.
- `paths.indexes_dir`: location of persisted vector indexes.
- `paths.manifests_dir`: location of source and index manifests.
- `paths.quality_dir`: location of the quality report.
- `llamaindex_version`: LlamaIndex runtime version used when building the persisted indexes. This should match `index_manifest.json`.
- `vector_store_type`: current vector store backend.
- `persist_backend`: persistence mode for saved indexes.

## 4. `cleaned_pages.jsonl`

Each line is one cleaned page record.

Fields:

- `doc_id`: canonical document identifier.
- `corpus`: corpus scope identifier; the current package uses `main`.
- `title`: document title.
- `source_path`: canonical raw-PDF path.
- `source_url`: original source URL.
- `page`: 1-based page number in the PDF.
- `total_pages`: total pages in the PDF.
- `page_type`: page label such as `abstract`, `main_content`, `references`, or `appendix`.
- `section_path`: best-effort section path for the dominant content on that page.
- `text`: cleaned page text.
- `token_count`: exact `cl100k_base` token count for `text`.
- `tokenizer_name`: tokenizer used to compute `token_count`.
- `excluded_from_index`: whether this page is excluded from chunking and indexing.

Complete example:

```json
{
  "doc_id": "engineering_rag_systems_2025",
  "corpus": "main",
  "title": "Engineering RAG Systems for Real-World Applications: Design, Development, and Evaluation",
  "source_path": "data/raw/main/engineering_rag_systems_2025.pdf",
  "source_url": "https://arxiv.org/abs/2506.20869",
  "page": 5,
  "total_pages": 16,
  "page_type": "main_content",
  "section_path": "3 Study Design / 3.1 Implementing RAG Systems",
  "text": "[FIGURE: Overview of the research methodology]\n– Generation Phase: The retrieved text chunks are concatenated with the original user query and passed into a large language model (LLM), such as GPT-4o, LLaMA 2 Uncensored, or Poro-34B, to synthesize contextually relevant responses. This approach improves factual accuracy, minimizes hallucinations, and delivers insights that are well aligned with domain-specific needs. Core Components. Each RAG-based system comprises multiple core components: – Data Sources: Knowledge bases include structured and unstructured documents, such as websites, municipal records, cybersecurity reports, agricultural research papers, engineering documents, and clinical guidelines. – Vector Database: The retrieved knowledge is stored as vector embeddings in FAISS, Pinecone, or OpenAI’s Vector Store, depending on the system’s latency and scalability requirements.",
  "token_count": 180,
  "tokenizer_name": "cl100k_base",
  "excluded_from_index": false
}
```

## 5. `chunks_xxx.jsonl`

Each line is one chunk record built from cleaned pages.

Fields:

- `chunk_id`: chunk identifier, unique within the corpus.
- `doc_id`: source document identifier.
- `corpus`: corpus scope.
- `title`: source document title.
- `section_path`: section path inherited from the chunk start location.
- `page_start`: first PDF page covered by the chunk.
- `page_end`: last PDF page covered by the chunk.
- `chunk_size`: target chunk size configured for the file variant.
- `overlap`: overlap used by the sliding window.
- `text`: chunk text.
- `token_count`: exact `cl100k_base` token count of `text`.
- `tokenizer_name`: tokenizer used to compute `token_count`.
- `excluded_from_index`: always `false` for persisted chunk files in the current main corpus.

Complete example:

```json
{
  "chunk_id": "policy_document_rag_2026_chunk_012",
  "doc_id": "policy_document_rag_2026",
  "corpus": "main",
  "title": "Chunking, Retrieval, and Re-ranking: An Empirical Evaluation of RAG Architectures for Policy Document Question Answering",
  "section_path": "C. Mathematical Formulation",
  "page_start": 3,
  "page_end": 3,
  "chunk_size": 256,
  "overlap": 26,
  "text": ".35 0.62 0.80 0.45 0.70 0.80\nB. Qualitative Case Study\nTo better illustrate the mechanism of failure and recovery, Table II provides a direct comparison of the outputs for Question 1. The Vanilla model provides a generic, medically accurate but policy-irrelevant definition. The Advanced RAG model successfully retrieves the specific CDC “reframing” requirement.",
  "token_count": 87,
  "tokenizer_name": "cl100k_base",
  "excluded_from_index": false
}
```

## 6. How To Load Vector Indexes

### Installation

Recommended approach with a standard virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Workspace-local dependency mode used in this repository:

```bash
pip install --target .deps -r requirements.txt
```

Why `.deps` exists:

- This project uses `.deps` as a local package directory to avoid polluting the global Python environment.
- All project scripts prepend `.deps` to `sys.path` if the directory exists.
- If teammates prefer a normal virtual environment, that also works. `.deps` is not required when packages are already installed in the active interpreter.

### Python Example

```python
from pathlib import Path
import sys

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".deps"))

from llama_index.core import StorageContext, load_index_from_storage
from scripts.embedding_utils import TransformersBgeEmbedding

embed_model = TransformersBgeEmbedding(model_name="BAAI/bge-base-en-v1.5")

def load_index(index_dir: str):
    storage_context = StorageContext.from_defaults(
        persist_dir=str((ROOT / index_dir).resolve())
    )
    return load_index_from_storage(
        storage_context=storage_context,
        embed_model=embed_model,
    )

index_256 = load_index("data/indexes/index_main_256")
index_512 = load_index("data/indexes/index_main_512")

retriever_256 = index_256.as_retriever(similarity_top_k=3)
retriever_512 = index_512.as_retriever(similarity_top_k=3)

query = "What is the purpose of CRAG's retrieval evaluator?"

results_256 = retriever_256.retrieve(query)
results_512 = retriever_512.retrieve(query)

for item in results_256:
    print(item.node.node_id, item.node.metadata["doc_id"], item.score)

for item in results_512:
    print(item.node.node_id, item.node.metadata["doc_id"], item.score)
```

Important note:

- Downstream code must import `TransformersBgeEmbedding` from `scripts.embedding_utils`.
- Do not assume the standard `llama_index.embeddings.huggingface.HuggingFaceEmbedding` path is used in this workspace.

## 7. `corpus` Field

Current meaning:

- `main`: official experimental corpus used for chunking, indexing, evaluation, and reporting.

Rules:

- All reported experiments must use `corpus = "main"`.
- Current persisted indexes only cover `main`.

## 8. `excluded_from_index` Field

Purpose:

- Controls whether a cleaned page is allowed to enter the chunking and indexing pipeline.

Current rules:

- `table_of_contents` pages are excluded.
- `references` pages are excluded.
- `appendix` pages are kept by default unless identified as high-noise appendix pages during cleaning.

Pipeline behavior:

- `scripts/chunk.py` only chunks records where `excluded_from_index = false`.
- `scripts/build_index.py` only loads chunk records where `excluded_from_index = false`.
- Current chunk files and indexes therefore contain only indexable content.

## 9. `index_manifest.json`

Current fields:

- `index_name`: persisted index directory name.
- `corpus_scope`: corpus used to build the index.
- `chunk_size`: target chunk size used for the source chunk file.
- `overlap`: overlap used in chunk generation.
- `tokenizer_name`: tokenizer used in the chunk pipeline.
- `embedding_model`: embedding model name used during index construction.
- `llamaindex_version`: actual runtime version used to build the index.
- `vector_store_type`: vector backend used by the index.
- `doc_count`: number of unique source documents covered by the index.
- `chunk_count`: number of indexed chunks.
- `build_command`: exact command used to build the index.

Use cases:

- sanity-checking which index matches which experiment,
- reporting build provenance in the project report,
- reproducing the exact index build command later.

## 10. Script Usage

### Current downstream pipeline

1. `scripts/chunk.py`
   - Purpose: build chunk-level JSONL outputs from `cleaned_pages.jsonl`.
   - Key args: `--config`, `--input`, `--output-dir`, `--chunk-size`.
   - Default behavior: builds all configured chunk variants.

2. `scripts/build_index.py`
   - Purpose: build persisted LlamaIndex indexes from chunk files.
   - Key args: `--config`, `--corpus`, `--chunk-size`.
   - Current default corpus: `main`.

3. `scripts/quality_check.py`
   - Purpose: verify chunk-level statistics and write `data_quality_report.md`.
   - Key args: `--config`, `--corpus`, `--chunk-size`.

4. `scripts/embedding_utils.py`
   - Purpose: expose `TransformersBgeEmbedding`, required for both index building and index loading.

### Recommended run order

Current GitHub handoff workflow:

```bash
python scripts/chunk.py --config config.yaml
python scripts/build_index.py --corpus main --chunk-size 256 --chunk-size 512
python scripts/quality_check.py --corpus main --chunk-size 256 --chunk-size 512
```

This handoff package starts from the cleaned corpus in `data/cleaned/cleaned_pages.jsonl`. Raw-paper selection and page-level cleaning have already been completed for the current baseline.

## 11. Recommended Downstream Workflow

Suggested continuation for the other teammates:

1. Start from `README.md` and `docs/interface_guide.md` to understand the current scope and artifact layout.
2. Load `index_main_256` and `index_main_512` using `scripts.embedding_utils.TransformersBgeEmbedding`.
3. Build retrieval and answer-generation pipelines on top of the main paper corpus only.
4. Use `data/eval/eval_seed_set.jsonl` as the first evaluation batch.

## 12. FAQ

**Q1. Do I need the raw PDFs to run retrieval experiments?**  
No. For retrieval, evaluation, and interface integration, the main dependencies are `data/chunks/*`, `data/indexes/*`, `data/manifests/index_manifest.json`, and `scripts/embedding_utils.py`.

**Q2. Why do I need `scripts.embedding_utils.TransformersBgeEmbedding` when loading an index?**  
Because the current workspace uses a custom embedding wrapper around `BAAI/bge-base-en-v1.5` for environment compatibility. Use the same wrapper at load time for consistent query embeddings.

**Q3. Are demo indexes available?**  
Not in the current baseline. Only `index_main_256` and `index_main_512` are built.

**Q4. Why do `source_path` fields point to `data/raw/main/...`?**  
That is the canonical raw-data location for the current baseline. The raw PDFs have been synchronized into `data/raw/main/` so metadata and files now match.

**Q5. What should the model team use by default?**  
Use `index_main_256` and `index_main_512`, import `TransformersBgeEmbedding`, and keep all experiments restricted to `corpus = "main"`.
