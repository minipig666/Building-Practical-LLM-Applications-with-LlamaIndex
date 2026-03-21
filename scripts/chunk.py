#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import tiktoken


DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


@dataclass(frozen=True)
class PageSpan:
    page: int
    section_path: str | None
    start_char: int
    end_char: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chunk-level JSONL outputs from cleaned page records")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--input", type=Path, default=None, help="Override cleaned page JSONL path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override chunk output directory")
    parser.add_argument(
        "--chunk-size",
        type=int,
        action="append",
        default=None,
        help="Only generate the specified chunk size(s). Can be passed multiple times.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a mapping at the top level")
    return data


def resolve_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_cleaned_pages(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("excluded_from_index"):
                continue
            records.append(record)
    return records


def build_document_text(doc_pages: list[dict[str, Any]]) -> tuple[str, list[PageSpan]]:
    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0
    separator = "\n\n"

    for page_index, page in enumerate(doc_pages):
        text = page["text"]
        start_char = cursor
        parts.append(text)
        cursor += len(text)
        spans.append(
            PageSpan(
                page=page["page"],
                section_path=page.get("section_path"),
                start_char=start_char,
                end_char=cursor,
            )
        )
        if page_index < len(doc_pages) - 1:
            parts.append(separator)
            cursor += len(separator)

    return "".join(parts), spans


def span_overlap(start: int, end: int, page_span: PageSpan) -> int:
    return max(0, min(end, page_span.end_char) - max(start, page_span.start_char))


def map_tokens_to_pages(offsets: list[int], text_length: int, page_spans: list[PageSpan]) -> list[int]:
    if not page_spans:
        return []

    token_pages: list[int] = []
    page_index = 0

    for token_index, start_char in enumerate(offsets):
        end_char = offsets[token_index + 1] if token_index + 1 < len(offsets) else text_length

        while page_index + 1 < len(page_spans) and page_spans[page_index].end_char <= start_char:
            page_index += 1

        best_index = page_index
        best_overlap = span_overlap(start_char, end_char, page_spans[page_index])

        lookahead = page_index + 1
        while lookahead < len(page_spans) and page_spans[lookahead].start_char < end_char:
            overlap = span_overlap(start_char, end_char, page_spans[lookahead])
            if overlap > best_overlap:
                best_index = lookahead
                best_overlap = overlap
            lookahead += 1

        if best_overlap == 0 and page_index + 1 < len(page_spans):
            best_index = page_index + 1

        token_pages.append(page_spans[best_index].page)

    return token_pages


def is_whitespace_only_token(token_id: int, encoding: tiktoken.Encoding) -> bool:
    token_text = encoding.decode([token_id])
    return bool(token_text) and token_text.isspace()


def trim_chunk_boundaries(token_ids: list[int], token_pages: list[int], encoding: tiktoken.Encoding) -> tuple[list[int], list[int]]:
    start_index = 0
    end_index = len(token_ids)

    while start_index < end_index and is_whitespace_only_token(token_ids[start_index], encoding):
        start_index += 1
    while end_index > start_index and is_whitespace_only_token(token_ids[end_index - 1], encoding):
        end_index -= 1

    return token_ids[start_index:end_index], token_pages[start_index:end_index]


def finalize_chunk(
    token_ids: list[int],
    token_pages: list[int],
    chunk_size: int,
    encoding: tiktoken.Encoding,
) -> tuple[str, int, list[int]]:
    normalized_ids = list(token_ids)
    normalized_pages = list(token_pages)

    while normalized_ids:
        chunk_text = encoding.decode(normalized_ids).strip()
        if not chunk_text:
            normalized_ids.pop()
            normalized_pages.pop()
            continue

        actual_token_count = len(encoding.encode(chunk_text))
        if actual_token_count <= chunk_size:
            return chunk_text, actual_token_count, normalized_pages

        normalized_ids.pop()
        normalized_pages.pop()

    return "", 0, []


def build_chunks_for_document(
    doc_pages: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    tokenizer_name: str,
    encoding: tiktoken.Encoding,
) -> list[dict[str, Any]]:
    if not doc_pages:
        return []

    if overlap >= chunk_size:
        raise ValueError(f"overlap must be smaller than chunk_size, got {overlap} >= {chunk_size}")

    full_text, page_spans = build_document_text(doc_pages)
    if not full_text.strip():
        return []

    token_ids = encoding.encode(full_text)
    if not token_ids:
        return []

    decoded_text, offsets = encoding.decode_with_offsets(token_ids)
    if decoded_text != full_text:
        raise ValueError(f"Decoded text mismatch for document {doc_pages[0]['doc_id']}")

    token_pages = map_tokens_to_pages(offsets, len(full_text), page_spans)
    page_lookup = {page["page"]: page for page in doc_pages}
    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 1
    stride = chunk_size - overlap

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_token_ids, chunk_token_pages = trim_chunk_boundaries(token_ids[start:end], token_pages[start:end], encoding)
        if not chunk_token_ids:
            start += stride
            continue

        chunk_text, actual_token_count, chunk_token_pages = finalize_chunk(
            token_ids=chunk_token_ids,
            token_pages=chunk_token_pages,
            chunk_size=chunk_size,
            encoding=encoding,
        )
        if not chunk_text:
            start += stride
            continue

        page_start = chunk_token_pages[0]
        page_end = chunk_token_pages[-1]
        first_page = page_lookup[page_start]
        last_page = page_lookup[page_end]

        chunks.append(
            {
                "chunk_id": f'{doc_pages[0]["doc_id"]}_chunk_{chunk_index:03d}',
                "doc_id": doc_pages[0]["doc_id"],
                "corpus": doc_pages[0]["corpus"],
                "title": doc_pages[0]["title"],
                "section_path": first_page.get("section_path"),
                "page_start": first_page["page"],
                "page_end": last_page["page"],
                "chunk_size": chunk_size,
                "overlap": overlap,
                "text": chunk_text,
                "token_count": actual_token_count,
                "tokenizer_name": tokenizer_name,
                "excluded_from_index": False,
            }
        )
        chunk_index += 1

        if end >= len(token_ids):
            break

        start += stride

    return chunks


def select_chunk_configs(configs: list[dict[str, Any]], requested_sizes: list[int] | None) -> list[dict[str, Any]]:
    if not requested_sizes:
        return configs

    selected = [config for config in configs if config["chunk_size"] in requested_sizes]
    missing = sorted(set(requested_sizes) - {config["chunk_size"] for config in selected})
    if missing:
        raise ValueError(f"Requested chunk size(s) not found in config.yaml: {missing}")
    return selected


def group_pages_by_document(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in sorted(records, key=lambda item: (item["corpus"], item["doc_id"], item["page"])):
        key = (record["corpus"], record["doc_id"])
        grouped.setdefault(key, []).append(record)
    return grouped


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def print_stats(chunk_size: int, output_path: Path, chunks: list[dict[str, Any]]) -> None:
    token_counts = [chunk["token_count"] for chunk in chunks]
    average_tokens = mean(token_counts) if token_counts else 0.0
    max_tokens = max(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0

    print(f"Chunk size {chunk_size}")
    print(f"  Output: {output_path}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Average tokens: {average_tokens:.2f}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Min tokens: {min_tokens}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)

    tokenizer_name = config["tokenizer_name"]
    chunk_configs = select_chunk_configs(config["chunk_configs"], args.chunk_size)
    cleaned_path = args.input or resolve_path(config_path.parent, config["paths"]["cleaned_pages"])
    output_dir = args.output_dir or resolve_path(config_path.parent, config["paths"]["chunks_dir"])

    if cleaned_path is None or output_dir is None:
        raise ValueError("Both cleaned_pages and chunks_dir must be configured")
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned page file not found: {cleaned_path}")

    encoding = tiktoken.get_encoding(tokenizer_name)
    records = load_cleaned_pages(cleaned_path)
    grouped_pages = group_pages_by_document(records)

    for chunk_config in chunk_configs:
        chunk_size = int(chunk_config["chunk_size"])
        overlap = int(chunk_config["overlap"])
        chunks: list[dict[str, Any]] = []

        for _, doc_pages in grouped_pages.items():
            chunks.extend(
                build_chunks_for_document(
                    doc_pages=doc_pages,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    tokenizer_name=tokenizer_name,
                    encoding=encoding,
                )
            )

        output_path = output_dir / f"chunks_{chunk_size}.jsonl"
        write_jsonl(output_path, chunks)
        print_stats(chunk_size, output_path, chunks)


if __name__ == "__main__":
    main()
