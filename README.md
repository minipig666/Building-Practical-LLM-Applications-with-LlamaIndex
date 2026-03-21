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
