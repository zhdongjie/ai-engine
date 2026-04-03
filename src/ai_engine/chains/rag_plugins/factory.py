# src/ai_engine/rag_plugins/factory.py
from typing import List
from .base import BaseRAGPlugin
from .i18n_plugin import I18nInstructionPlugin
from .java_docs_plugin import JavaDocsPlugin

PLUGIN_REGISTRY = {
    "java_tutor": [
        I18nInstructionPlugin(),
        JavaDocsPlugin(),
    ],
    "normal_chat": [
        I18nInstructionPlugin()
    ],
    "session_title": [
        I18nInstructionPlugin()
    ],
}

def get_rag_plugins(biz_type: str) -> List[BaseRAGPlugin]:
    """根据业务类型，获取对应的 RAG 后处理插件实例列表。"""
    return PLUGIN_REGISTRY.get(biz_type, [])