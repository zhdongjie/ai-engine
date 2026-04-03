from ai_engine.infra.embedding.factory import get_embedding_provider
from ai_engine.infra.vector_store.factory import get_vector_provider


class VectorStoreManager:
    def __init__(self):
        self._embedding_provider = None
        self._provider = None

    @property
    def embedding_provider(self):
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    @property
    def embeddings(self):
        """给 LangChain 用"""
        return self.embedding_provider.raw

    @property
    def provider(self):
        if self._provider is None:
            self._provider = get_vector_provider(self.embedding_provider)
        return self._provider

    @property
    def store(self):
        return self.provider.vector_store


# 暴露单例对象供全局调用
vdb_manager = VectorStoreManager()
