#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import tiktoken


DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
DEFAULT_CORPUS = "main"
REQUIRED_FIELDS = [
    "chunk_id",
    "doc_id",
    "corpus",
    "title",
    "section_path",
    "page_start",
    "page_end",
    "chunk_size",
    "overlap",
    "text",
    "token_count",
    "tokenizer_name",
    "excluded_from_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chunk-level data quality checks and write a Markdown report")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument(
        "--chunk-size",
        type=int,
        action="append",
        default=None,
        help="Only check the specified chunk size(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="Only check the specified corpus scope(s). Defaults to main.",
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
            if record.get("corpus") != corpus:
                continue
            records.append(record)
    return records


def summarize_records(records: list[dict[str, Any]], tokenizer_name: str) -> dict[str, Any]:
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    token_counts: list[int] = []
    doc_counter: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    empty_chunks = 0
    duplicate_counter: Counter[str] = Counter()
    cross_page_chunks = 0
    excluded_true = 0
    token_mismatch_count = 0
    corpus_counter: Counter[str] = Counter()

    for record in records:
        corpus_counter[str(record.get("corpus"))] += 1
        if record.get("excluded_from_index"):
            excluded_true += 1
        if record.get("page_end", 0) > record.get("page_start", 0):
            cross_page_chunks += 1

        text = record.get("text", "")
        if not str(text).strip():
            empty_chunks += 1
        duplicate_counter[str(text).strip()] += 1

        for field in REQUIRED_FIELDS:
            value = record.get(field)
            if value is None or (isinstance(value, str) and not value.strip() and field not in {"section_path"}):
                missing_fields[field] += 1

        actual_token_count = len(tokenizer.encode(str(text)))
        token_counts.append(actual_token_count)
        if actual_token_count != record.get("token_count"):
            token_mismatch_count += 1

        doc_counter[str(record.get("doc_id"))] += 1

    duplicate_chunks = sum(count - 1 for text, count in duplicate_counter.items() if text and count > 1)
    doc_chunk_counts = list(doc_counter.values())

    return {
        "total_chunks": len(records),
        "doc_count": len(doc_counter),
        "avg_tokens": statistics.mean(token_counts) if token_counts else 0.0,
        "median_tokens": statistics.median(token_counts) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "empty_chunks": empty_chunks,
        "duplicate_chunks": duplicate_chunks,
        "cross_page_chunks": cross_page_chunks,
        "excluded_true": excluded_true,
        "token_mismatch_count": token_mismatch_count,
        "missing_fields": dict(sorted((field, count) for field, count in missing_fields.items() if count > 0)),
        "doc_chunk_min": min(doc_chunk_counts) if doc_chunk_counts else 0,
        "doc_chunk_max": max(doc_chunk_counts) if doc_chunk_counts else 0,
        "doc_chunk_avg": statistics.mean(doc_chunk_counts) if doc_chunk_counts else 0.0,
        "corpus_counter": dict(corpus_counter),
    }


def render_report(
    summaries: list[dict[str, Any]],
    tokenizer_name: str,
    embedding_model: str,
    vector_store_type: str,
) -> str:
    lines: list[str] = []
    lines.append("# Data Quality Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Tokenizer: `{tokenizer_name}`")
    lines.append(f"- Embedding model baseline: `{embedding_model}`")
    lines.append(f"- Vector store backend: `{vector_store_type}`")
    lines.append("")

    for summary in summaries:
        lines.append(f"## {summary['corpus']} / chunk_size={summary['chunk_size']}")
        lines.append("")
        lines.append(f"- Total chunks: `{summary['total_chunks']}`")
        lines.append(f"- Documents covered: `{summary['doc_count']}`")
        lines.append(f"- Average tokens: `{summary['avg_tokens']:.2f}`")
        lines.append(f"- Median tokens: `{summary['median_tokens']}`")
        lines.append(f"- Min tokens: `{summary['min_tokens']}`")
        lines.append(f"- Max tokens: `{summary['max_tokens']}`")
        lines.append(f"- Cross-page chunks: `{summary['cross_page_chunks']}`")
        lines.append(f"- Empty chunks: `{summary['empty_chunks']}`")
        lines.append(f"- Duplicate chunks: `{summary['duplicate_chunks']}`")
        lines.append(f"- Token count mismatches: `{summary['token_mismatch_count']}`")
        lines.append(f"- Chunks marked excluded_from_index=true: `{summary['excluded_true']}`")
        lines.append(
            f"- Chunks per document: min `{summary['doc_chunk_min']}`, max `{summary['doc_chunk_max']}`, avg `{summary['doc_chunk_avg']:.2f}`"
        )
        lines.append(f"- Corpus values seen: `{summary['corpus_counter']}`")
        if summary["missing_fields"]:
            lines.append(f"- Missing metadata fields: `{summary['missing_fields']}`")
        else:
            lines.append("- Missing metadata fields: none")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    paths = config["paths"]

    chunk_configs = select_chunk_configs(config["chunk_configs"], args.chunk_size)
    corpora = args.corpus or [DEFAULT_CORPUS]

    chunks_dir = resolve_path(config_path.parent, paths["chunks_dir"])
    quality_dir = resolve_path(config_path.parent, paths.get("quality_dir"))
    if chunks_dir is None or quality_dir is None:
        raise ValueError("chunks_dir and quality_dir must be configured")

    summaries: list[dict[str, Any]] = []

    for corpus in corpora:
        for chunk_config in chunk_configs:
            chunk_size = int(chunk_config["chunk_size"])
            chunk_path = chunks_dir / f"chunks_{chunk_size}.jsonl"
            if not chunk_path.exists():
                raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

            records = load_chunk_records(chunk_path, corpus=corpus)
            summary = summarize_records(records, tokenizer_name=config["tokenizer_name"])
            summary["chunk_size"] = chunk_size
            summary["corpus"] = corpus
            summaries.append(summary)

            print(f"Checked corpus={corpus}, chunk_size={chunk_size}")
            print(f"  Total chunks: {summary['total_chunks']}")
            print(f"  Avg tokens: {summary['avg_tokens']:.2f}")
            print(f"  Duplicate chunks: {summary['duplicate_chunks']}")
            print(f"  Token count mismatches: {summary['token_mismatch_count']}")

    report = render_report(
        summaries=summaries,
        tokenizer_name=config["tokenizer_name"],
        embedding_model=config["embedding_model"],
        vector_store_type=config.get("vector_store_type", "SimpleVectorStore"),
    )

    quality_dir.mkdir(parents=True, exist_ok=True)
    report_path = quality_dir / "data_quality_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Quality report written to {report_path}")


if __name__ == "__main__":
    main()
