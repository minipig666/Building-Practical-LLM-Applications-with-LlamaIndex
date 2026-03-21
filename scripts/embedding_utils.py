#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import torch
from transformers import AutoModel, AutoTokenizer
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr


class TransformersBgeEmbedding(BaseEmbedding):
    model_name: str
    max_length: int = 512

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _device: str = PrivateAttr()
    _dimension: int = PrivateAttr()
    _query_instruction: str = PrivateAttr(
        default="Represent this sentence for searching relevant passages: "
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._device = "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()
        self._dimension = int(getattr(self._model.config, "hidden_size"))

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * attention_mask).sum(dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.cpu().tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._encode_batch([self._query_instruction + query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._encode_batch([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._encode_batch(texts)
