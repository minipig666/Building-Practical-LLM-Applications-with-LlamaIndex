import gradio as gr
from pathlib import Path
import sys
import json
from typing import List, Dict, Any
import re
from collections import Counter

ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".deps"))


# ============================================
# 基于本地 JSONL 文件的检索器（无需模型）
# ============================================
class LocalChunkRetriever:
    """
    从项目的 chunks_256.jsonl 直接加载数据
    使用 TF-IDF 风格的关键词检索，不需要任何 embedding 模型
    """

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
        """从 JSONL 文件加载所有片段"""
        with open(self.chunk_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        self.chunks.append(chunk)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 第 {line_num} 行解析失败: {e}")

    def _build_index(self):
        """构建简单的倒排索引（词 -> 包含该词的片段列表）"""
        self.inverted_index = {}

        for idx, chunk in enumerate(self.chunks):
            text = chunk.get("text", "")
            # 简单的分词：字母数字 + 下划线，转小写
            words = re.findall(r'\b[a-z0-9_]+\b', text.lower())

            for word in set(words):  # 每个片段内去重
                if word not in self.inverted_index:
                    self.inverted_index[word] = []
                self.inverted_index[word].append(idx)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        基于关键词匹配的检索
        返回包含相关片段及元数据
        """
        if not query or not query.strip():
            return []

        # 对查询进行分词
        query_words = re.findall(r'\b[a-z0-9_]+\b', query.lower())
        if not query_words:
            return []

        # 计算每个片段的得分（词频累计）
        scores = Counter()
        for word in query_words:
            if word in self.inverted_index:
                for chunk_idx in self.inverted_index[word]:
                    # 基础得分：每匹配一个词得 1 分
                    scores[chunk_idx] += 1

                    # 额外加分：如果词出现在标题中
                    chunk = self.chunks[chunk_idx]
                    title = chunk.get("title", "").lower()
                    if word in title:
                        scores[chunk_idx] += 2

        # 获取得分最高的 top_k 个片段
        top_chunk_indices = [idx for idx, _ in scores.most_common(top_k)]

        results = []
        for idx in top_chunk_indices:
            chunk = self.chunks[idx]
            results.append({
                "node": chunk,
                "score": scores[idx] / (len(query_words) + 1e-6)  # 归一化得分
            })

        return results


# ============================================
# 初始化检索器（使用你的真实数据）
# ============================================
try:
    # 使用 256-token 的片段（你也可以改成 512）
    retriever = LocalChunkRetriever(chunk_size=256)
    print("✅ 检索器初始化成功")
except Exception as e:
    print(f"❌ 检索器初始化失败: {e}")
    raise


# ============================================
# RAG 查询函数
# ============================================
def rag_query(question: str):
    """
    处理用户问题，返回检索结果和来源
    """
    if not question or not question.strip():
        return "请输入问题", "无来源"

    try:
        # 执行检索
        nodes_with_score = retriever.retrieve(question, top_k=3)

        if not nodes_with_score:
            return "未找到相关内容，请尝试其他问题", "无匹配结果"

        # 构建结果展示
        result = f"✅ 已从 {len(retriever.chunks)} 个片段中检索到相关内容\n\n"
        sources = ""

        for i, item in enumerate(nodes_with_score):
            chunk = item["node"]
            score = item["score"]

            doc_id = chunk.get("doc_id", "未知文档")
            title = chunk.get("title", "无标题")
            page_start = chunk.get("page_start", "?")
            page_end = chunk.get("page_end", "?")
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", "")
            token_count = chunk.get("token_count", 0)

            # 截断过长的文本用于展示
            display_text = text[:300] + "..." if len(text) > 300 else text

            result += f"【片段 {i + 1} | 相关度 {score:.2f}】\n"
            result += f"{display_text}\n\n"

            sources += f"{'=' * 60}\n"
            sources += f"【片段 {i + 1}】\n"
            sources += f"文档ID: {doc_id}\n"
            sources += f"标题: {title}\n"
            sources += f"片段ID: {chunk_id}\n"
            sources += f"页码: {page_start}~{page_end}\n"
            sources += f"Token数: {token_count}\n"
            sources += f"相关度: {score:.3f}\n"
            sources += f"内容:\n{text}\n\n"

        return result, sources

    except Exception as e:
        import traceback
        error_msg = f"查询出错: {str(e)}\n{traceback.format_exc()}"
        return error_msg, ""


# ============================================
# 构建 Gradio 界面
# ============================================
def create_demo():
    """创建 Gradio 界面"""
    # 移除 theme 参数（根据警告，移到 launch() 中）
    with gr.Blocks(title="RAG 论文助手") as demo:
        gr.Markdown("""
        # 📄 RAG 论文问答助手

        **基于本地论文数据的检索系统** | 无需网络 | 无需模型下载

        当前知识库包含以下论文：
        - Engineering RAG Systems (2025)
        - Corrective Retrieval-Augmented Generation (2024)
        - RAGCheckER (2024)
        - RAFT: Domain-Specific RAG (2024)
        - 以及其他 RAG 相关论文
        """)

        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Textbox(
                    label="💬 输入问题",
                    placeholder="例如: CRAG 的检索评估器有什么作用？",
                    lines=4,
                    show_label=True
                )

                with gr.Row():
                    btn = gr.Button("🔍 检索", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ 清空", variant="secondary")

                gr.Markdown("""
                ### 📌 示例问题
                - RAG 系统如何减少幻觉？
                - DPR 相比 BM25 有什么优势？
                - 什么是检索增强的语言模型预训练？
                - CRAG 的纠正机制是如何工作的？
                """)

            with gr.Column(scale=2):
                answer = gr.Textbox(
                    label="📚 检索结果",
                    lines=12,
                    show_label=True,
                    interactive=False
                )

        # 移除 show_copy_button 参数（Gradio 新版本不支持）
        sources = gr.Textbox(
            label="📖 完整来源信息",
            lines=15,
            interactive=False
        )

        # 绑定事件
        btn.click(
            rag_query,
            inputs=question,
            outputs=[answer, sources]
        )

        clear_btn.click(
            lambda: ("", "", ""),
            outputs=[question, answer, sources]
        )

        # 示例问题快捷输入
        gr.Examples(
            examples=[
                ["RAG 系统如何减少模型幻觉？"],
                ["CRAG 的检索评估器有什么作用？"],
                ["什么是密集段落检索（DPR）？"],
                ["REALM 和 RAG 有什么区别？"],
            ],
            inputs=question,
            label="点击示例快速测试"
        )

    return demo


# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动 RAG 论文问答助手")
    print(f"📁 项目根目录: {ROOT}")
    print(f"📊 数据文件: {ROOT / 'data/chunks/chunks_256.jsonl'}")
    print("=" * 60)

    demo = create_demo()

    # 将 theme 参数移到 launch() 中
    demo.launch(
        server_port=7865,
        share=False,  # 如需外网访问可设为 True
        show_error=True,
        quiet=False,
        theme=gr.themes.Soft()  # 主题移到此处
    )