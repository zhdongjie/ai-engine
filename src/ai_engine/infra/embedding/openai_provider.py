# src/ai_engine/infra/embedding/openai_provider.py
from typing import List

from ai_engine.infra.embedding.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):

    def __init__(self):
        # TODO 适配OpenAIEmbeddings
        # OpenAIEmbeddings(
        #     api_key=settings.OPENAI_API_KEY.get_secret_value(),
        #     base_url=settings.OPENAI_API_BASE,
        #     model=settings.OPENAI_EMBEDDING_MODEL,
        # )
        self._client = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)

    @property
    def raw(self):
        """给 LangChain 用"""
        return self._client