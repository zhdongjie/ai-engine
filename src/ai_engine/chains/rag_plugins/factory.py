# src/ai_engine/rag_plugins/factory.py
from typing import List

from .base import BaseRAGPlugin
from .java_docs_plugin import JavaDocsPlugin

_PLUGIN_REGISTRY = {
    "java_tutor": [
        JavaDocsPlugin(),
    ],
    "normal_chat": [],
}


def get_rag_plugins(biz_type: str) -> List[BaseRAGPlugin]:
    """
    根据业务类型，获取对应的 RAG 后处理插件列表。
    如果未配置，则返回空列表，保证外部的 for 循环安全执行。
    """
    return _PLUGIN_REGISTRY.get(biz_type, [])
