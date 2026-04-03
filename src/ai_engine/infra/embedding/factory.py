# src/ai_engine/infra/embedding/factory.py
from ai_engine.core.settings import settings


def get_embedding_provider():
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "openai":
        from ai_engine.infra.embedding.openai_provider import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider()

    elif provider == "qwen":
        from ai_engine.infra.embedding.qwen_provider import QwenEmbeddingProvider
        return QwenEmbeddingProvider()

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")