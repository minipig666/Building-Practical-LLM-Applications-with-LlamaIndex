# engine.py
from llama_index.core import StorageContext, load_index_from_storage
import config

class RAGAppEngine:
    def __init__(self, index_dir=config.DEFAULT_INDEX_DIR):
        self.index_dir = index_dir
        self.index = self.load_index()

    def load_index(self):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
            index = load_index_from_storage(storage_context)
            print(f"✅ 索引加载成功: {self.index_dir}")
            return index
        except Exception as e:
            print(f"❌ 索引加载失败: {e}")
            return None

    def query(self, question, similarity_top_k=config.SIMILARITY_TOP_K):
        if not self.index:
            return "Error: Index not loaded."
        query_engine = self.index.as_query_engine(similarity_top_k=similarity_top_k)
        response = query_engine.query(question)
        return response