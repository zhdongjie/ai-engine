# src/ai_engine/infra/vector_store/factory.py
from ai_engine.core.settings import settings
from .chroma_provider import ChromaProvider
from .pg_provider import PGVectorProvider


class VectorStoreFactory:
    @staticmethod
    def get_provider(embeddings):
        v_type = settings.VECTOR_STORE_TYPE.lower()
        if v_type == "postgresql":
            return PGVectorProvider(embeddings)
        elif v_type == "chroma":
            return ChromaProvider(embeddings)
        else:
            raise ValueError(f"不支持的向量库类型: {v_type}")

vector_manager = VectorStoreFactory()