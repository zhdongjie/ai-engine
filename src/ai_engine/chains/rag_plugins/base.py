# src/ai_engine/rag_plugins/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any


class BaseRAGPlugin(ABC):
    """RAG 后处理插件基类"""

    @abstractmethod
    def process(self, docs: List[Any], context: str, extra: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        处理检索结果
        :param docs: 检索召回的 Document 列表
        :param context: 已经拼接好的上下文纯文本
        :param extra: 需要传递给大模型或前端的额外结构化数据
        :return: (处理后的 context, 处理后的 extra)
        """
        pass