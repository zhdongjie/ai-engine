# src/ai_engine/infra/llm/llm_factory.py
from dashscope import TextReRank
from dashscope.api_entities.dashscope_response import ReRankResponse
from langchain_openai import ChatOpenAI

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings


class LLMFactory:
    """全球统一的大模型与外部 AI 服务生产车间"""

    @staticmethod
    def get_model(yaml_config: dict = None, **runtime_overrides) -> ChatOpenAI:
        """
        基于三级优先级动态拼装大模型实例：
        Base (全局兜底) < YAML (业务级) < Runtime (节点级动态覆写)
        """
        yaml_config = yaml_config or {}

        base_settings = {
            "api_key": settings.QWEN_API_KEY.get_secret_value(),
            "base_url": settings.QWEN_API_BASE,
            "model": settings.QWEN_MODEL_LLM,
            "temperature": settings.TEMPERATURE,
            "max_retries": 3,
        }

        final_params = base_settings | yaml_config | runtime_overrides

        return ChatOpenAI(**final_params)

    @staticmethod
    def call_rerank(query: str, documents: list) -> ReRankResponse:
        """统一封装的重排服务调用"""
        try:
            return TextReRank.call(
                model=settings.QWEN_MODEL_RERANK,
                query=query,
                documents=documents,
                top_n=settings.RERANK_TOP_N,
                api_key=settings.QWEN_API_KEY.get_secret_value(),
            )
        except Exception as e:
            logger.error(f"全局 Rerank 服务调用异常: {e}")
            raise
