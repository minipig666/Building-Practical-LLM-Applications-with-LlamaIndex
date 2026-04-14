import gradio as gr
from pathlib import Path
import sys
import json
from typing import List, Dict, Any
import re
from collections import Counter

ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT))


from manager import ModelDeploymentManager
from engine import RAGAppEngine
import config


model_manager = ModelDeploymentManager()


class LocalChunkRetriever:
    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size
        self.chunks: List[Dict] = []
        self.chunk_file = ROOT / f"data/chunks/chunks_{chunk_size}.jsonl"

        if not self.chunk_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.chunk_file}")

        self._load_chunks()
        self._build_index()
        print(f"✅ Loaded {len(self.chunks)} {chunk_size}-token chunks")

    def _load_chunks(self):
        with open(self.chunk_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        self.chunks.append(chunk)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Failed to parse line {line_num}: {e}")

    def _build_index(self):
        self.inverted_index = {}
        for idx, chunk in enumerate(self.chunks):
            text = chunk.get("text", "")
            words = re.findall(r'\b[a-z0-9_]+\b', text.lower())
            for word in set(words):
                if word not in self.inverted_index:
                    self.inverted_index[word] = []
                self.inverted_index[word].append(idx)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        query_words = re.findall(r'\b[a-z0-9_]+\b', query.lower())
        if not query_words:
            return []

        scores = Counter()
        for word in query_words:
            if word in self.inverted_index:
                for chunk_idx in self.inverted_index[word]:
                    scores[chunk_idx] += 1
                    chunk = self.chunks[chunk_idx]
                    title = chunk.get("title", "").lower()
                    if word in title:
                        scores[chunk_idx] += 2

        top_chunk_indices = [idx for idx, _ in scores.most_common(top_k)]
        results = []
        for idx in top_chunk_indices:
            chunk = self.chunks[idx]
            results.append({
                "node": chunk,
                "score": scores[idx] / (len(query_words) + 1e-6)
            })
        return results



def rag_query(question: str, model_choice: str, chunk_size: int):
    if not question or not question.strip():
        return "Please enter a question", "No sources"

    try:
        # ----------------------
        # 1. Switch model
        # ----------------------
        model_manager.set_llm_backend(model_choice)

        # ----------------------
        # 2. Load corresponding chunk index
        # ----------------------
        index_dir = config.INDEX_PATH_256 if chunk_size == 256 else config.INDEX_PATH_512
        engine = RAGAppEngine(index_dir=index_dir)

        # ----------------------
        # 3. Dynamic Top-K
        # ----------------------
        top_k = 1 if model_choice == "t5" else 3

        # ----------------------
        # 4. Execute RAG query
        # ----------------------
        response = engine.query(question, similarity_top_k=top_k)

        # ----------------------
        # 5. Show local retrieval (keep original functionality)
        # ----------------------
        retriever = LocalChunkRetriever(chunk_size=chunk_size)
        nodes_with_score = retriever.retrieve(question, top_k=3)

        # ----------------------
        # Assemble results
        # ----------------------
        result = f"🤖 Model Response ({model_choice.upper()}):\n{str(response)}\n\n"
        sources = ""

        for i, item in enumerate(nodes_with_score):
            chunk = item["node"]
            doc_id = chunk.get("doc_id", "?")
            title = chunk.get("title", "Untitled")
            text = chunk.get("text", "")
            sources += f"===== Source {i+1} =====\nDocument: {title}\nContent: {text}\n\n"

        return result, sources

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}", ""



def create_demo():
    with gr.Blocks(title="RAG Paper Assistant") as demo:
        gr.Markdown("""
        # 📄 RAG Paper Q&A Assistant
        **Local Paper Data + Dual Model Retrieval**
        """)

        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Textbox(
                    label="💬 Enter Question",
                    placeholder="e.g.: What is the role of CRAG's retriever evaluator?",
                    lines=3
                )

                # New: Model selection
                model_choice = gr.Radio(
                    ["mistral", "t5"],
                    value="mistral",
                    label="Select Model"
                )

                # New: Chunk size selection
                chunk_size = gr.Radio(
                    [256, 512],
                    value=256,
                    label="Chunk Size"
                )

                with gr.Row():
                    btn = gr.Button("🔍 Retrieve", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear")

            with gr.Column(scale=2):
                answer = gr.Textbox(label="📚 Answer", lines=15, interactive=False)

        sources = gr.Textbox(label="📖 Full Sources", lines=12, interactive=False)

        # Bind events
        btn.click(
            rag_query,
            inputs=[question, model_choice, chunk_size],
            outputs=[answer, sources]
        )

        clear_btn.click(
            lambda: ("", "", "", ""),
            outputs=[question, answer, sources]
        )

    return demo


# ============================================
# Launch
# ============================================
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_port=7865,
        share=False,
        theme=gr.themes.Soft()
    )