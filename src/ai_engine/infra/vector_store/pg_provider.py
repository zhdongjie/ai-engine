# src/ai_engine/infra/vector_store/pg_provider.py
from langchain_postgres import PGVector
from sqlalchemy import text
from ai_engine.infra.db.pgsql import db_manager
from ai_engine.core.logger import logger
from .base import BaseVectorStoreProvider


class PGVectorProvider(BaseVectorStoreProvider):
    def __init__(self, embeddings):
        self._embeddings = embeddings
        self._vs = None  # 初始设为 None，不立即初始化

    @property
    def vector_store(self):

        if self._vs is None:
            self._vs = PGVector(
                embeddings=self._embeddings,
                collection_name="ai_knowledge_base",
                connection=db_manager.engine,
                use_jsonb=True,
                create_extension=False,
            )
        return self._vs

    def clear_all(self):
        """物理清空向量表"""
        logger.warning("正在物理清空 PostgreSQL 向量表...")
        # 注意：这里直接操作底层连接，不依赖 self.vector_store
        with db_manager.engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS langchain_pg_collection CASCADE"))

        self._vs = None

    def delete_by_path_md5(self, path_md5: str):
        """按路径指纹删除旧切片"""
        with db_manager.engine.begin() as conn:
            query = text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'path_md5' = :p_md5")
            conn.execute(query, {"p_md5": path_md5})

    def add_documents(self, docs):
        """执行数据写入"""
        # 使用 self.vector_store 触发懒加载检查
        self.vector_store.add_documents(docs)