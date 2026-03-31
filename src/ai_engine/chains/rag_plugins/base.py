# src/ai_engine/rag_plugins/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

from langchain_core.runnables import RunnableConfig


class BaseRAGPlugin(ABC):
    """RAG 后处理插件基类：支持对检索结果、上下文及额外数据的二次加工"""

    @abstractmethod
    def process(
            self,
            docs: List[Any],
            context: str,
            extra: Dict[str, Any],
            config: RunnableConfig
    ) -> Tuple[str, Dict[str, Any]]:
        """
        :param docs: 检索召回的 Document 列表（包含 metadata）
        :param context: 已初步拼接的上下文文本
        :param extra: 传递给 LLM 或前端的额外数据（如 sources, scores）
        :param config: 执行配置（包含 configurable.lang, metadata 等）
        :return: (加工后的 context, 加工后的 extra)
        """
        pass
