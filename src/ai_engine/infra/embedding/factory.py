# src/ai_engine/infra/embedding/factory.py
from typing import Callable, Any
from ai_engine.core.settings import settings


# 延迟导入的闭包函数
def _get_openai_provider():
    from ai_engine.infra.embedding.openai_provider import OpenAIEmbeddingProvider
    return OpenAIEmbeddingProvider()


def _get_qwen_provider():
    from ai_engine.infra.embedding.qwen_provider import QwenEmbeddingProvider
    return QwenEmbeddingProvider()


# 统一的注册表
EMBEDDING_REGISTRY: dict[str, Callable[[], Any]] = {
    "openai": _get_openai_provider,
    "qwen": _get_qwen_provider,
}


def get_embedding_provider():
    provider_name = settings.EMBEDDING_PROVIDER.lower()
    factory_method = EMBEDDING_REGISTRY.get(provider_name)

    if not factory_method:
        raise ValueError(f"Unsupported embedding provider: {provider_name}")

    return factory_method()