import gradio as gr

# 演示用的假函数
def fake_answer(question):
    answer = """这是演示回答，模型正在开发中。
系统会基于 10 篇 RAG 论文为你提供准确、可溯源的回答。"""

    source = """【片段 1｜相关性 0.92】
RAG（Retrieval-Augmented Generation）通过检索外部知识增强生成效果…

【片段 2｜相关性 0.88】
向量数据库用于存储文档嵌入，实现快速语义检索…"""

    return answer, source

# ======================
# 优化版漂亮布局
# ======================
with gr.Blocks(title="RAG 论文问答助手", theme=gr.themes.Soft()) as demo:
    # 顶部标题
    gr.Markdown("""
# 📄 RAG 论文问答助手
**基于 10 篇 arXiv 论文 | 检索增强生成问答系统**
""")

    # 使用行布局：左右分栏
    with gr.Row():
        # 左侧：问题输入
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 输入问题")
            question = gr.Textbox(
                label="",
                placeholder="请输入你想查询的 RAG 相关问题…",
                lines=3
            )
            btn = gr.Button("🚀 开始查询", variant="primary", size="lg")

        # 右侧：回答
        with gr.Column(scale=2):
            gr.Markdown("### 📝 系统回答")
            answer = gr.Textbox(label="", lines=5)

    # 占满整行的来源片段区域
    gr.Markdown("### 🔗 来源文献片段（可溯源）")
    sources = gr.Textbox(label="", lines=10, interactive=False)

    # 按钮绑定
    btn.click(
        fn=fake_answer,
        inputs=question,
        outputs=[answer, sources]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)