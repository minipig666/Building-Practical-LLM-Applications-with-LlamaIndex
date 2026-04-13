import gradio as gr
from pathlib import Path
import sys
import json
from typing import List, Dict, Any
import re
from collections import Counter

ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT))

# ========================
# 导入我们的模型 & 引擎
# ========================
from manager import ModelDeploymentManager
from engine import RAGAppEngine
import config

# 初始化全局模型管理器
model_manager = ModelDeploymentManager()

# ============================================
# 本地 JSONL 检索器（保留你原有代码）
# ============================================
class LocalChunkRetriever:
    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size
        self.chunks: List[Dict] = []
        self.chunk_file = ROOT / f"data/chunks/chunks_{chunk_size}.jsonl"

        if not self.chunk_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.chunk_file}")

        self._load_chunks()
        self._build_index()
        print(f"✅ 已加载 {len(self.chunks)} 个 {chunk_size}-token 片段")

    def _load_chunks(self):
        with open(self.chunk_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        self.chunks.append(chunk)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 第 {line_num} 行解析失败: {e}")

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


# ============================================
# 【升级】RAG 查询：支持模型 + 分块切换
# ============================================
def rag_query(question: str, model_choice: str, chunk_size: int):
    if not question or not question.strip():
        return "请输入问题", "无来源"

    try:
        # ----------------------
        # 1. 切换模型
        # ----------------------
        model_manager.set_llm_backend(model_choice)

        # ----------------------
        # 2. 加载对应分块索引
        # ----------------------
        index_dir = config.INDEX_PATH_256 if chunk_size == 256 else config.INDEX_PATH_512
        engine = RAGAppEngine(index_dir=index_dir)

        # ----------------------
        # 3. 动态 Top-K
        # ----------------------
        top_k = 1 if model_choice == "t5" else 3

        # ----------------------
        # 4. 执行 RAG 问答
        # ----------------------
        response = engine.query(question, similarity_top_k=top_k)

        # ----------------------
        # 5. 同时展示本地检索（保留你原有功能）
        # ----------------------
        retriever = LocalChunkRetriever(chunk_size=chunk_size)
        nodes_with_score = retriever.retrieve(question, top_k=3)

        # ----------------------
        # 组装展示结果
        # ----------------------
        result = f"🤖 模型回答 ({model_choice.upper()}):\n{str(response)}\n\n"
        sources = ""

        for i, item in enumerate(nodes_with_score):
            chunk = item["node"]
            doc_id = chunk.get("doc_id", "?")
            title = chunk.get("title", "无标题")
            text = chunk.get("text", "")
            sources += f"===== 来源 {i+1} =====\n文档：{title}\n内容：{text}\n\n"

        return result, sources

    except Exception as e:
        import traceback
        return f"错误：{str(e)}\n{traceback.format_exc()}", ""


# ============================================
# Gradio 界面（保留你原有风格）
# ============================================
def create_demo():
    with gr.Blocks(title="RAG 论文助手") as demo:
        gr.Markdown("""
        # 📄 RAG 论文问答助手
        **基于本地论文数据 + 双模型检索**
        """)

        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Textbox(
                    label="💬 输入问题",
                    placeholder="例如: CRAG 的检索评估器有什么作用？",
                    lines=3
                )

                # 新增：模型切换
                model_choice = gr.Radio(
                    ["mistral", "t5"],
                    value="mistral",
                    label="选择模型"
                )

                # 新增：分块切换
                chunk_size = gr.Radio(
                    [256, 512],
                    value=256,
                    label="分块大小"
                )

                with gr.Row():
                    btn = gr.Button("🔍 检索", variant="primary")
                    clear_btn = gr.Button("🗑️ 清空")

            with gr.Column(scale=2):
                answer = gr.Textbox(label="📚 回答结果", lines=15, interactive=False)

        sources = gr.Textbox(label="📖 完整来源", lines=12, interactive=False)

        # 绑定事件
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
# 启动
# ============================================
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_port=7865,
        share=False,
        theme=gr.themes.Soft()
    )