# src/ai_engine/infra/db/vdb.py
import logging

from langchain_openai import OpenAIEmbeddings

from ai_engine.core.settings import settings
from ai_engine.infra.db.pgsql import db_manager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    全局向量数据库管理器 (单例)
    负责统一初始化 Embeddings 和 VectorStore，并对业务层屏蔽底层引擎差异
    """

    def __init__(self):
        self._embeddings = None
        self._vector_store = None

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """懒加载初始化全局统一的 Embedding 模型"""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                api_key=settings.QWEN_API_KEY.get_secret_value(),
                base_url=settings.QWEN_API_BASE,
                model=settings.QWEN_MODEL_EMBEDDING,
                check_embedding_ctx_length=False
            )
            logger.info(f"全局 Embedding 模型 [{settings.QWEN_MODEL_EMBEDDING}] 初始化完成")
        return self._embeddings

    @property
    def store(self):
        """懒加载初始化向量数据库引擎，支持 PGVector 与 Chroma 动态切换"""
        if self._vector_store is None:
            v_type = settings.VECTOR_STORE_TYPE.lower()

            if v_type == "postgresql":
                from langchain_postgres import PGVector
                self._vector_store = PGVector(
                    embeddings=self.embeddings,
                    collection_name="ai_knowledge_base",
                    connection=db_manager.engine,
                    use_jsonb=True,
                    create_extension=False,
                )
                logger.info("PGVector 向量数据库引擎已挂载")
            else:
                from langchain_chroma import Chroma
                self._vector_store = Chroma(
                    persist_directory=settings.chroma_persist_dir,
                    embedding_function=self.embeddings
                )
                logger.info("Chroma 向量数据库引擎已挂载")

        return self._vector_store


# 暴露单例对象供全局调用
vdb_manager = VectorStoreManager()
