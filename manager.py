# manager.py 最终版（768维）
import torch
from transformers import pipeline
from llama_index.core import Settings
from llama_index.core.llms import CustomLLM, CompletionResponse
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.base.llms.types import LLMMetadata
from typing import Any

from scripts.embedding_utils import TransformersBgeEmbedding
import config

class FlanT5LLM(CustomLLM):
    pipe: Any = None

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt = f"Context: {prompt}\nAnswer only based on context: "
        outputs = self.pipe(prompt, truncation=True, max_new_tokens=128)
        return CompletionResponse(text=outputs[0]["generated_text"].strip())

    @property
    def metadata(self):
        return LLMMetadata(context_window=512, num_output=128, model_name=config.T5_MODEL)

class ModelDeploymentManager:
    def __init__(self):
        self.device = config.DEVICE
        self._current_model_type = None
        self.t5_pipe = None

        # ✅ 强制使用 768 维模型，和索引完全一致
        self.embed_model = TransformersBgeEmbedding(
            model_name="BAAI/bge-base-en-v1.5"
        )
        Settings.embed_model = self.embed_model
        print(">> ✅ Embedding 加载成功 | 维度：768")

    def set_llm_backend(self, model_type="mistral"):
        if self._current_model_type == model_type:
            return Settings.llm

        if model_type == "mistral":
            from llama_index.llms.ollama import Ollama
            llm = Ollama(model=config.MISTRAL_MODEL)
            print(">> Backend: Mistral-7B")

        elif model_type == "t5":
            if not self.t5_pipe:
                self.t5_pipe = pipeline("text2text-generation", model=config.T5_MODEL, device_map="auto")
            llm = FlanT5LLM(pipe=self.t5_pipe)
            print(">> Backend: FLAN-T5-Base")

        self._current_model_type = model_type
        Settings.llm = llm
        return llm