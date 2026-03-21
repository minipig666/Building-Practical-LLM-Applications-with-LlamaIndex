#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from scripts.embedding_utils import TransformersBgeEmbedding


DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
DEFAULT_CORPUS = "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LlamaIndex vector indexes from chunk JSONL files")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument(
        "--chunk-size",
        type=int,
        action="append",
        default=None,
        help="Only build the specified chunk size(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="Only build the specified corpus scope(s). Defaults to main.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a top-level mapping")
    return data


def resolve_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def select_chunk_configs(configs: list[dict[str, Any]], requested_sizes: list[int] | None) -> list[dict[str, Any]]:
    if not requested_sizes:
        return configs

    selected = [config for config in configs if int(config["chunk_size"]) in requested_sizes]
    missing = sorted(set(requested_sizes) - {int(config["chunk_size"]) for config in selected})
    if missing:
        raise ValueError(f"Requested chunk size(s) not found in config.yaml: {missing}")
    return selected


def load_chunk_records(path: Path, corpus: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("excluded_from_index"):
                continue
            if record.get("corpus") != corpus:
                continue
            records.append(record)
    return records


def build_nodes(records: list[dict[str, Any]]) -> list[TextNode]:
    nodes: list[TextNode] = []
    for record in records:
        metadata = {
            "chunk_id": record["chunk_id"],
            "doc_id": record["doc_id"],
            "corpus": record["corpus"],
            "title": record["title"],
            "section_path": record["section_path"],
            "page_start": record["page_start"],
            "page_end": record["page_end"],
            "chunk_size": record["chunk_size"],
            "overlap": record["overlap"],
            "tokenizer_name": record["tokenizer_name"],
        }
        nodes.append(
            TextNode(
                id_=record["chunk_id"],
                text=record["text"],
                metadata=metadata,
                excluded_embed_metadata_keys=list(metadata.keys()),
            )
        )
    return nodes


def build_command(config_path: Path, corpus: str, chunk_size: int) -> str:
    config_arg = config_path.name if config_path.parent == ROOT else str(config_path)
    return f"python scripts/build_index.py --config {config_arg} --corpus {corpus} --chunk-size {chunk_size}"


def upsert_manifest(manifest_path: Path, new_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing: list[dict[str, Any]] = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, list):
                existing = payload

    by_name = {record["index_name"]: record for record in existing if isinstance(record, dict) and "index_name" in record}
    for record in new_records:
        by_name[record["index_name"]] = record

    merged = sorted(by_name.values(), key=lambda item: (item["corpus_scope"], item["chunk_size"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=True, indent=2)
    return merged


def build_index_for_corpus(
    chunk_records: list[dict[str, Any]],
    corpus: str,
    chunk_size: int,
    overlap: int,
    config_path: Path,
    indexes_dir: Path,
    embedding_model_name: str,
    tokenizer_name: str,
    vector_store_type: str,
) -> dict[str, Any]:
    if not chunk_records:
        raise ValueError(f"No chunk records found for corpus={corpus}, chunk_size={chunk_size}")

    index_name = f"index_{corpus}_{chunk_size}"
    persist_dir = indexes_dir / index_name
    persist_dir.mkdir(parents=True, exist_ok=True)

    embed_model = TransformersBgeEmbedding(model_name=embedding_model_name, embed_batch_size=16)
    nodes = build_nodes(chunk_records)

    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    index.storage_context.persist(persist_dir=str(persist_dir))

    print(f"Built {index_name}")
    print(f"  Loaded chunks: {len(chunk_records)}")
    print(f"  Index dimension: {embed_model.dimension}")
    print(f"  Persist path: {persist_dir}")

    return {
        "index_name": index_name,
        "corpus_scope": corpus,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "tokenizer_name": tokenizer_name,
        "embedding_model": embedding_model_name,
        "llamaindex_version": version("llama-index-core"),
        "vector_store_type": vector_store_type,
        "doc_count": len({record["doc_id"] for record in chunk_records}),
        "chunk_count": len(chunk_records),
        "build_command": build_command(config_path=config_path, corpus=corpus, chunk_size=chunk_size),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)

    chunk_configs = select_chunk_configs(config["chunk_configs"], args.chunk_size)
    corpora = args.corpus or [DEFAULT_CORPUS]
    paths = config["paths"]
    chunks_dir = resolve_path(config_path.parent, paths["chunks_dir"])
    indexes_dir = resolve_path(config_path.parent, paths["indexes_dir"])
    manifests_dir = resolve_path(config_path.parent, paths["manifests_dir"])

    if chunks_dir is None or indexes_dir is None or manifests_dir is None:
        raise ValueError("chunks_dir, indexes_dir, and manifests_dir must be configured")

    embedding_model_name = config["embedding_model"]
    tokenizer_name = config["tokenizer_name"]
    vector_store_type = config.get("vector_store_type", "SimpleVectorStore")

    manifest_records: list[dict[str, Any]] = []

    for corpus in corpora:
        for chunk_config in chunk_configs:
            chunk_size = int(chunk_config["chunk_size"])
            overlap = int(chunk_config["overlap"])
            chunk_path = chunks_dir / f"chunks_{chunk_size}.jsonl"
            if not chunk_path.exists():
                raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

            records = load_chunk_records(chunk_path, corpus=corpus)
            manifest_records.append(
                build_index_for_corpus(
                    chunk_records=records,
                    corpus=corpus,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    config_path=config_path,
                    indexes_dir=indexes_dir,
                    embedding_model_name=embedding_model_name,
                    tokenizer_name=tokenizer_name,
                    vector_store_type=vector_store_type,
                )
            )

    manifest_path = manifests_dir / "index_manifest.json"
    upsert_manifest(manifest_path, manifest_records)
    print(f"Index manifest written to {manifest_path}")


if __name__ == "__main__":
    main()

# Example:
# from pathlib import Path
# from llama_index.core import StorageContext, load_index_from_storage
# from scripts.embedding_utils import TransformersBgeEmbedding
#
# embed_model = TransformersBgeEmbedding(model_name="BAAI/bge-base-en-v1.5")
# storage_context = StorageContext.from_defaults(
#     persist_dir=str(Path("data/indexes/index_main_256").resolve())
# )
# index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)
# retriever = index.as_retriever(similarity_top_k=3)
# results = retriever.retrieve("What are the differences between CRAG and RAFT?")
