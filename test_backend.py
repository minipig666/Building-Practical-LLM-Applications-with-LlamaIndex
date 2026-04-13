


# test_backend.py 纯后端 RAG 测试｜无页面｜无前端｜可选择块大小 256/512
import sys
sys.path.append(".")

from manager import ModelDeploymentManager
from engine import RAGAppEngine

# ==============================================
# 👉 在这里选择块大小：256 或 512
CHUNK_SIZE = 512  # 可选：256 或 512
# ==============================================

# 自动匹配你真实的索引路径
index_dir_map = {
    256: "./data/indexes/index_main_256",
    512: "./data/indexes/index_main_512"
}

# 确认使用的索引目录
selected_index_dir = index_dir_map[CHUNK_SIZE]
print(f"📚 使用 RAG 论文索引 | 块大小：{CHUNK_SIZE}")
print(f"📂 索引路径：{selected_index_dir}\n")

# ====================== 初始化模型 ======================
print("⏳ 加载 Embedding 模型 & LLM 模型...")
manager = ModelDeploymentManager()
#manager.set_llm_backend("mistral")  # 使用 mistral 本地模型
manager.set_llm_backend("t5")     # 用本地T5

# ====================== 加载 RAG 引擎 ======================
rag_engine = RAGAppEngine(index_dir=selected_index_dir)

# ====================== 提问（RAG 论文问题） ======================
question = "请介绍这个 RAG 项目的数据集包含哪些内容？"
# question = "RAG 项目使用了哪些块大小？"
# question = "这个项目清理了多少篇 arXiv 论文？"

print(f"\n🔍 问题：{question}")

# 发起查询
response = rag_engine.query(question)

print("\n✅ 回答：")
print(response)