# src/ai_engine/infra/vector_store/factory.py
from ai_engine.core.settings import settings
from ai_engine.infra.vector_store.base import BaseVectorStoreProvider


def _get_pg_provider(embeddings):
    from .pg_provider import PGVectorProvider
    return PGVectorProvider(embeddings)


def _get_chroma_provider(embeddings):
    from .chroma_provider import ChromaProvider
    return ChromaProvider(embeddings)


VECTOR_STORE_REGISTRY = {
    "postgresql": _get_pg_provider,
    "chroma": _get_chroma_provider,
}


def get_vector_provider(embeddings) -> BaseVectorStoreProvider:
    v_type = settings.VECTOR_STORE_TYPE.lower()
    factory_method = VECTOR_STORE_REGISTRY.get(v_type)

    if not factory_method:
        raise ValueError(f"不支持的向量库类型: {v_type}")

    return factory_method(embeddings)