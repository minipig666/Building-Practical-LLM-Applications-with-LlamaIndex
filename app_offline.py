import gradio as gr
from pathlib import Path
import sys
import json
from typing import List, Dict, Any
import re
from collections import Counter

ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT))

# ============================================
# 基于本地 JSONL 文件的检索器（无需模型、无需联网）
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
# 初始化离线检索器（完全本地，不联网！）
# ============================================
retriever = LocalChunkRetriever(chunk_size=256)
print("✅ 检索器初始化成功（本地模式，无联网）")


# ============================================
# 本地 RAG 查询（纯离线，不调用任何模型）
# ============================================
def rag_query(question: str):
    if not question or not question.strip():
        return "请输入问题", "无来源"

    try:
        nodes_with_score = retriever.retrieve(question, top_k=3)

        if not nodes_with_score:
            return "未找到相关内容，请尝试其他问题", "无匹配结果"

        result = f"✅ 本地检索完成（无模型、无联网）\n\n"
        sources = ""

        for i, item in enumerate(nodes_with_score):
            chunk = item["node"]
            score = item["score"]
            title = chunk.get("title", "无标题")
            text = chunk.get("text", "")
            display_text = text[:300] + "..." if len(text) > 300 else text

            result += f"【片段 {i+1} | 相关度 {score:.2f}】\n{display_text}\n\n"
            sources += f"===== 来源 {i+1} =====\n标题：{title}\n内容：{text}\n\n"

        return result, sources

    except Exception as e:
        import traceback
        return f"错误：{str(e)}", ""


# ============================================
# Gradio 界面
# ============================================
def create_demo():
    with gr.Blocks(title="RAG 论文助手（本地离线版）") as demo:
        gr.Markdown("""
        # 📄 RAG 论文问答助手（本地离线版）
        **无需联网 | 无需模型 | 直接运行**
        """)

        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Textbox(label="输入问题", placeholder="例如：什么是RAG？", lines=3)
                btn = gr.Button("🔍 检索", variant="primary")
                clear_btn = gr.Button("清空")

            with gr.Column(scale=2):
                answer = gr.Textbox(label="检索结果", lines=15, interactive=False)

        sources = gr.Textbox(label="完整来源", lines=12, interactive=False)

        btn.click(rag_query, inputs=question, outputs=[answer, sources])
        clear_btn.click(lambda: ("", "", ""), outputs=[question, answer, sources])

    return demo


# ============================================
# 启动（完全离线！）
# ============================================
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_port=7865)