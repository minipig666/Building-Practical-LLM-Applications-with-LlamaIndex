# rebuild_index.py
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from manager import ModelDeploymentManager
import config

# 初始化模型（768维）
model_manager = ModelDeploymentManager()

# 重建索引
def rebuild_index():
    print("🔨 开始重建索引...")
    documents = SimpleDirectoryReader("data/raw/main").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=config.INDEX_PATH_256)
    index.storage_context.persist(persist_dir=config.INDEX_PATH_512)
    print("✅ 索引重建完成！维度统一 768！")

if __name__ == "__main__":
    rebuild_index()