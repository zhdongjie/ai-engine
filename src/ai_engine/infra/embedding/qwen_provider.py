# src/ai_engine/infra/embedding/qwen_provider.py
from typing import List

from openai import OpenAI

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.embedding.base import EmbeddingProvider


class QwenEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self._client = QwenEmbeddings(
            api_key=settings.QWEN_API_KEY.get_secret_value(),
            base_url=settings.QWEN_API_BASE,
            model=settings.QWEN_MODEL_EMBEDDING,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)

    @property
    def raw(self):
        return self._client


class QwenEmbeddings:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        处理 Qwen API 的 10 条批次限制
        """
        # 1. 深度清洗：过滤 None、空字符串、纯空格
        texts = [str(t) for t in texts if t and str(t).strip()]

        if not texts:
            return []

        # 2. 分批处理：10 条一组
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            logger.info(
                f"正在向 Qwen API 请求 Embedding: 批次 {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                # 按照原始顺序提取向量
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Qwen Embedding 批次请求失败: {e}")
                raise e

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        # query 通常只有一条，直接调用即可，但仍需确保非空
        if not text or not text.strip():
            # 返回一个零向量或根据业务抛出异常
            logger.warning("接收到空查询，无法生成 Embedding")
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
